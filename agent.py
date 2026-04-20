"""
Legal Document Assistant — Capstone Project
Agentic AI Course 2026 | Dr. Kanthi Kiran Sirra

Agent with:
  1. LangGraph StateGraph (8 nodes)
  2. ChromaDB RAG (12 legal documents)
  3. MemorySaver + thread_id (conversational memory)
  4. Self-reflection eval node (faithfulness scoring)
  5. Tool use (web search via DuckDuckGo + datetime)
  6. Streamlit deployment (see capstone_streamlit.py)
"""

import os
import datetime
from typing import TypedDict, Annotated
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME          = "llama-3.3-70b-versatile"
EMBED_MODEL         = "all-MiniLM-L6-v2"
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES    = 2
TOP_K               = 3

# ─────────────────────────────────────────────
# PART 1 — KNOWLEDGE BASE (12 legal documents)
# ─────────────────────────────────────────────
LEGAL_DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Contract Law — Essential Elements",
        "text": (
            "A valid contract requires four essential elements: offer, acceptance, consideration, "
            "and intention to create legal relations. An offer is a clear, definite proposal made "
            "by the offeror to the offeree, which if accepted unconditionally creates a binding "
            "agreement. The offer must be communicated to the offeree and may be made to a specific "
            "person or to the world at large (as in reward cases). An offer can be revoked any time "
            "before acceptance, but not after. Acceptance must be unconditional, unequivocal, and "
            "communicated to the offeror — a counter-offer destroys the original offer entirely and "
            "creates a new offer. Silence generally cannot constitute acceptance. "
            "Consideration refers to something of value exchanged by each party — it is the price "
            "paid for the other party's promise. Consideration need not be adequate (i.e., equal in "
            "value) but must be sufficient in law, meaning it must have some recognisable legal value. "
            "Past consideration — something done before the promise was made — is generally not valid. "
            "Without consideration, a promise is unenforceable unless made by deed (a signed, witnessed "
            "written document). Both parties must also intend the agreement to be legally binding. "
            "Social and domestic agreements (e.g., between family members) are presumed NOT to be "
            "legally binding, whereas commercial agreements are presumed to BE binding. "
            "Capacity to contract is an additional requirement — minors (under 18), those of unsound "
            "mind, and intoxicated persons may lack full contractual capacity. Contracts with minors "
            "are generally voidable at the minor's option, though contracts for necessaries are binding. "
            "A contract formed without these essential elements is void (has no legal effect) or "
            "voidable (valid until one party chooses to set it aside)."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Contract Law — Breach and Remedies",
        "text": (
            "A breach of contract occurs when one party fails to perform their contractual obligations "
            "without lawful excuse. A breach may be actual (failure to perform at the due date) or "
            "anticipatory (one party clearly indicates in advance they will not perform — the innocent "
            "party may treat the contract as ended immediately and sue without waiting for the due date). "
            "The primary remedy for breach of contract is damages — a monetary award intended to put "
            "the innocent party in the position they would have been in had the contract been performed. "
            "The key types of damages are: compensatory damages (covering direct actual loss), "
            "consequential damages (for foreseeable indirect loss arising from special circumstances "
            "known to both parties), nominal damages (where a breach occurred but no financial loss "
            "resulted), punitive damages (rare in contract law, awarded to punish), and liquidated "
            "damages (pre-agreed sums written into the contract — enforceable if they represent a "
            "genuine pre-estimate of loss, not a penalty). "
            "Equitable remedies include specific performance (a court order requiring the breaching "
            "party to fulfil their exact obligations — commonly awarded for unique goods or land) and "
            "injunctions (court orders prohibiting certain conduct). Rescission sets aside the contract "
            "and restores both parties to their original positions. "
            "The innocent party has a strict duty to mitigate their loss — they cannot sit back and "
            "allow losses to accumulate unnecessarily. Failure to mitigate reduces the damages awarded. "
            "Limitation periods generally require contract claims to be brought within six years of "
            "the breach (twelve years for contracts made by deed). Claims brought after this period "
            "are time-barred and cannot be pursued in court."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Contract Law — Vitiating Factors",
        "text": (
            "Certain factors can vitiate (undermine) a contract, making it void or voidable. "
            "Misrepresentation occurs when a false statement of fact (not opinion) made by one party "
            "induces the other party to enter the contract. It may be fraudulent (made knowingly or "
            "recklessly), negligent (made without reasonable grounds for belief), or innocent (made "
            "in good faith) — each attracting different remedies ranging from rescission and damages "
            "to rescission alone. A statement of opinion or future intention is generally not a "
            "misrepresentation unless the maker did not genuinely hold that opinion. "
            "Mistake refers to a fundamental error by one or both parties at the time of contracting. "
            "Common mistake (both parties share the same fundamental mistake, e.g., both believe the "
            "subject matter exists when it does not) can render a contract void. Mutual mistake "
            "(parties are at cross-purposes) and unilateral mistake (only one party is mistaken) "
            "have more limited effects. "
            "Duress involves illegitimate pressure — physical threats or economic duress — that "
            "overrides a party's free will, making the contract voidable. The victim must have had "
            "no reasonable alternative but to agree. Undue influence arises where one party exploits "
            "a position of trust, confidence, or dominance over another — relationships such as "
            "solicitor-client, doctor-patient, and parent-child may raise a presumption of undue "
            "influence. Illegality renders a contract void if it involves commission of a crime, "
            "a civil wrong, or is contrary to public policy (e.g., contracts to commit fraud, "
            "restrain trade unreasonably, or oust court jurisdiction). A contract induced by any "
            "vitiating factor may be rescinded — setting it aside and returning both parties to "
            "their pre-contract position — subject to bars such as affirmation, lapse of time, "
            "third party rights, and impossibility of restitution."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Criminal Procedure — Arrest and Custody Rights",
        "text": (
            "An arrest is the deprivation of a person's liberty by a law enforcement officer or "
            "private citizen, based on reasonable suspicion that the person has committed, is "
            "committing, or is about to commit a cognisable offence. Under the Code of Criminal "
            "Procedure, 1973 (CrPC), a police officer may arrest without a warrant for cognisable "
            "offences; a warrant is required for non-cognisable offences. "
            "Upon arrest, the accused has the fundamental right to be informed of the grounds of "
            "arrest immediately and clearly, as guaranteed by Article 22(1) of the Indian Constitution. "
            "The accused has the right to remain silent — anything said can and may be used as "
            "evidence against them in court. The right to consult and be defended by a legal "
            "practitioner of their choice must be communicated without delay (Article 22(1)). "
            "An arrested person must be produced before the nearest magistrate within 24 hours of "
            "arrest, excluding travel time (Article 22(2) and Section 57 CrPC). Detention beyond "
            "24 hours requires judicial authorisation. "
            "Bail may be granted by the police officer in charge for bailable offences. For "
            "non-bailable offences, only a court may grant bail, considering factors such as the "
            "nature of the offence, the accused's antecedents, and the risk of flight or tampering "
            "with evidence. Anticipatory bail under Section 438 CrPC may be sought in anticipation "
            "of arrest. "
            "Custodial interrogation must not involve coercion, torture, threats, or inducements — "
            "such methods render any resulting confession inadmissible. Confessions made to a police "
            "officer while in police custody are inadmissible under Section 25 of the Indian Evidence "
            "Act, 1872. A confession made voluntarily before a magistrate under Section 164 CrPC is "
            "admissible. The D.K. Basu guidelines (Supreme Court, 1996) impose additional procedural "
            "safeguards on arrest and detention, including mandatory documentation and medical examination."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Criminal Procedure — Trial Process",
        "text": (
            "A criminal trial in India follows a structured process under the Code of Criminal "
            "Procedure, 1973. Trials are classified as Sessions trials (for serious offences), "
            "Warrant trials (for offences punishable with more than two years imprisonment), and "
            "Summons trials (for minor offences). "
            "The trial begins with the framing of charges — the court frames charges based on "
            "the police report (chargesheet) and documents submitted. The accused enters a plea; "
            "a guilty plea may allow the court to proceed directly to sentencing. If the accused "
            "pleads not guilty, the trial proceeds. "
            "The prosecution presents its case first, examining witnesses (examination-in-chief). "
            "The defence has the right to cross-examine each prosecution witness to test credibility "
            "and challenge evidence. The prosecution may re-examine witnesses after cross-examination "
            "to clarify matters arising from it. Documentary and physical evidence is exhibited and "
            "marked during this stage. "
            "After the prosecution closes its case, the court considers whether a prima facie case "
            "has been made out. If not, the accused may be acquitted at this stage. If yes, the "
            "accused is given an opportunity to enter their defence — they may examine witnesses "
            "and produce documents. "
            "Under Section 313 CrPC, the court puts incriminating evidence and circumstances "
            "directly to the accused for their explanation — the accused is not under oath at this "
            "stage and their answers may be considered but cannot be used as a confession. "
            "Both sides present closing arguments. The court then pronounces judgment, which must "
            "give reasons for conviction or acquittal. The accused is presumed innocent until proven "
            "guilty beyond reasonable doubt — this is the cornerstone of criminal justice. "
            "An acquittal by a competent court bars re-trial for the same offence under the double "
            "jeopardy protection (Article 20(2) of the Constitution and Section 300 CrPC). "
            "Appeals against conviction lie to the Sessions Court, High Court, and ultimately the "
            "Supreme Court depending on the sentence imposed."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Civil Procedure — Filing a Suit",
        "text": (
            "A civil suit is initiated by filing a plaint — a written statement by the plaintiff "
            "setting out the facts constituting the cause of action, the relief sought, and the "
            "valuation of the suit for the purpose of court fees and jurisdiction. The plaint must "
            "comply with Order VII of the Code of Civil Procedure, 1908 (CPC) and must disclose a "
            "cause of action — failure to do so renders the plaint liable to rejection under "
            "Order VII Rule 11. "
            "The suit must be filed in a court of competent jurisdiction, determined by three factors: "
            "subject matter jurisdiction (the nature of the dispute), pecuniary jurisdiction (the "
            "monetary value of the claim), and territorial jurisdiction (the geographic connection "
            "of the parties or cause of action to the court's area). "
            "Court fees must be paid at the time of filing based on the suit's valuation — "
            "underpayment may result in the plaint being returned for correction. "
            "After the plaint is filed and accepted, the court issues a summons to the defendant, "
            "requiring them to appear and file a written statement (defence) within 30 days, "
            "extendable by the court to a maximum of 90 days on sufficient cause shown. "
            "The written statement must specifically deny or admit each allegation in the plaint — "
            "evasive denials are treated as admissions. The defendant may also raise a counterclaim "
            "against the plaintiff in the written statement. "
            "Failure by the defendant to appear or file a written statement may result in an "
            "ex-parte decree against them — a decree passed without hearing the defendant. "
            "Parties may file interim applications at this stage: for temporary injunctions "
            "(to preserve the status quo), attachment before judgment (to secure assets), "
            "or appointment of a receiver. The court examines such applications on the basis of "
            "prima facie case, balance of convenience, and irreparable harm."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Civil Procedure — Discovery and Evidence",
        "text": (
            "Discovery is the pre-trial and trial process by which parties exchange information "
            "and documents relevant to the dispute, ensuring no party is taken by surprise at trial. "
            "Under Order XI of the CPC, a party may apply to the court for: discovery of documents "
            "(requiring the other party to disclose all relevant documents in their possession or "
            "power), inspection of documents (physically examining disclosed documents), and "
            "interrogatories (written questions that the other party must answer under oath). "
            "Documents must be disclosed if they are relevant to the issues in the case and not "
            "protected by privilege. Legal professional privilege (also called attorney-client "
            "privilege) protects confidential communications between a lawyer and their client made "
            "for the purpose of obtaining legal advice or in anticipation of litigation — these "
            "documents cannot be compelled to be produced. Without prejudice communications made "
            "in genuine settlement negotiations are also privileged. "
            "Documents admitted into evidence are exhibited and marked sequentially — Exhibit P-1, "
            "P-2 etc. for plaintiff's documents and Exhibit D-1, D-2 etc. for defendant's documents. "
            "Electronic records — emails, digital documents, computer printouts — are admissible "
            "under Section 65B of the Indian Evidence Act, 1872, provided a certificate of "
            "authenticity is furnished by the person responsible for the computer system. Failure "
            "to provide a Section 65B certificate renders electronic evidence inadmissible. "
            "In civil proceedings, witnesses give their evidence-in-chief by affidavit, followed "
            "by oral cross-examination in court. Expert witnesses may be appointed by the court "
            "or called by parties to give opinion evidence on technical, scientific, medical, or "
            "other specialised matters. The court is not bound by expert opinion but must consider "
            "it along with all other evidence."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Tort Law — Negligence",
        "text": (
            "Negligence is the most prevalent tort, imposing liability for careless conduct that "
            "causes harm to another. To establish negligence, a claimant must prove three elements "
            "on the balance of probabilities: a duty of care owed by the defendant to the claimant, "
            "breach of that duty, and damage caused by the breach. "
            "A duty of care exists where the defendant ought reasonably to have foreseen that "
            "their conduct could cause harm to the claimant — the neighbour principle established "
            "in Donoghue v Stevenson (1932). In India, the courts have broadly adopted this "
            "principle. The standard of care required is that of a reasonable person in the "
            "defendant's position — an objective standard, not based on the defendant's personal "
            "abilities or beliefs. In professional negligence cases (doctors, lawyers, engineers), "
            "the standard is that of a reasonably competent professional in that field — the Bolam "
            "test (Bolam v Friern Hospital Management Committee, 1957) is widely applied in "
            "medical negligence cases. "
            "Breach occurs when the defendant's conduct falls below the required standard of care. "
            "Factors relevant to breach include the probability of harm, the severity of potential "
            "harm, the cost of precautions, and the social utility of the defendant's conduct. "
            "Causation is established using the 'but for' test — but for the defendant's breach, "
            "would the harm have occurred? Where multiple causes contribute, the 'material "
            "contribution' test may apply. Remoteness limits liability — the defendant is only "
            "liable for types of harm that were reasonably foreseeable, not all consequences. "
            "Contributory negligence by the claimant reduces damages proportionately to their "
            "share of fault. Common defences include volenti non fit injuria (the claimant "
            "voluntarily assumed the risk of harm), contributory negligence, and inevitable "
            "accident (harm that could not have been avoided by reasonable care)."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Tort Law — Strict and Vicarious Liability",
        "text": (
            "Strict liability imposes responsibility for harm without requiring proof of fault, "
            "negligence, or intention. The foundational English case is Rylands v Fletcher (1868), "
            "where the House of Lords held that a person who brings onto their land anything "
            "likely to do mischief if it escapes must keep it at their peril — if they fail to "
            "do so, they are prima facie answerable for all the damage which is the natural "
            "consequence of its escape. The rule applies to non-natural uses of land involving "
            "dangerous things such as water, fire, chemicals, and explosives. Defences include "
            "act of God, consent of the claimant, default of the claimant, and act of a stranger. "
            "The Indian Supreme Court significantly extended strict liability in M.C. Mehta v "
            "Union of India (1987), establishing the doctrine of absolute liability for enterprises "
            "engaged in inherently hazardous or dangerous activities. Unlike Rylands v Fletcher, "
            "absolute liability admits of NO exceptions — it applies even where the escape is "
            "caused by an act of God or a third party. The enterprise is absolutely liable to "
            "compensate all those affected, and the larger and more profitable the enterprise, "
            "the greater the liability. This principle was applied in the Bhopal Gas Tragedy cases. "
            "Vicarious liability holds one person legally responsible for the torts committed by "
            "another, arising from their relationship. The most common relationship is employer "
            "and employee — an employer is vicariously liable for torts committed by employees "
            "in the course of their employment. The 'close connection' test (Lister v Hesley Hall, "
            "2001) determines whether the employee's wrongful act was so closely connected to "
            "their employment that it would be fair to hold the employer liable. "
            "Independent contractors generally do not attract vicarious liability, though exceptions "
            "exist where the employer retains control, the work is extra-hazardous, or the employer "
            "has non-delegable statutory duties."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Constitutional Rights — Fundamental Rights",
        "text": (
            "Part III of the Indian Constitution (Articles 12 to 35) guarantees Fundamental Rights "
            "that are enforceable against the State and, in some cases, against private individuals. "
            "Article 12 defines 'State' broadly to include the Government of India, State governments, "
            "Parliament, State Legislatures, and all local and other authorities within India or under "
            "the control of the Government of India. "
            "Article 14 guarantees equality before the law and equal protection of the laws to all "
            "persons — it prohibits arbitrary action and requires the State to treat equals equally "
            "and unequals unequally in proportion to their inequality. "
            "Article 19 protects six freedoms for citizens: speech and expression, peaceful assembly "
            "without arms, association, free movement throughout India, residence and settlement "
            "anywhere in India, and the right to practise any profession or carry on any occupation, "
            "trade, or business. Each freedom is subject to reasonable restrictions imposed by law "
            "on specified grounds (e.g., public order, morality, sovereignty of India). "
            "Article 21 protects the right to life and personal liberty — no person shall be deprived "
            "of their life or personal liberty except according to procedure established by law. The "
            "Supreme Court has expansively interpreted Article 21 to include the right to privacy "
            "(Justice K.S. Puttaswamy v Union of India, 2017), right to livelihood, right to health, "
            "right to education, right to a dignified life, right to a speedy trial, and right against "
            "solitary confinement. "
            "Article 22 provides specific protections against arbitrary arrest and detention. "
            "Articles 25 to 28 guarantee freedom of conscience and the right to freely profess, "
            "practise, and propagate religion, subject to public order, morality, and health. "
            "Articles 32 and 226 provide constitutional remedies — the Supreme Court and High Courts "
            "may issue writs (habeas corpus — to produce an illegally detained person; mandamus — to "
            "compel performance of a public duty; prohibition — to prevent inferior courts from "
            "exceeding jurisdiction; certiorari — to quash decisions; quo warranto — to challenge "
            "unlawful holding of public office) to enforce fundamental rights. "
            "Fundamental Rights may be suspended during a national emergency proclaimed under "
            "Article 352, except Articles 20 and 21, which can never be suspended."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Intellectual Property — Copyright and Trademark",
        "text": (
            "Copyright protects original literary, dramatic, musical, and artistic works, as well "
            "as cinematograph films and sound recordings. In India, copyright is governed by the "
            "Copyright Act, 1957, as amended. Originality requires that the work originate from "
            "the author and involve some minimal degree of creativity — mere effort or labour "
            "without creativity is insufficient. Copyright protection arises automatically upon "
            "creation of the work — no registration is required, though registration under "
            "Section 44 of the Act creates a rebuttable presumption of ownership and is useful "
            "in infringement proceedings. "
            "The duration of copyright is the lifetime of the author plus 60 years from the "
            "beginning of the calendar year following death. For cinematograph films, sound "
            "recordings, photographs, and works of government and international organisations, "
            "the term is 60 years from publication. "
            "Copyright infringement occurs when any act restricted to the copyright owner — "
            "reproduction, communication to the public, adaptation, translation — is done without "
            "a licence. Fair dealing is a statutory defence under Section 52 permitting limited "
            "use without permission for private research, criticism or review, reporting of current "
            "events, and education. The amount used, the purpose, and the effect on the market "
            "for the original work are all relevant factors. "
            "Trademark law protects distinctive signs — words, logos, shapes, colours, sounds — "
            "that distinguish the goods or services of one enterprise from those of others. In "
            "India, trademarks are governed by the Trade Marks Act, 1999. Registration grants "
            "the proprietor an exclusive right to use the mark for the registered goods/services "
            "for an initial period of 10 years, renewable indefinitely for further 10-year periods. "
            "Infringement under Section 29 occurs when an identical or deceptively similar mark "
            "is used in the course of trade for identical or similar goods or services, causing "
            "a likelihood of confusion or association in the minds of the public. "
            "Passing off is the common law action available for unregistered marks — it protects "
            "against misrepresentation by a trader that their goods or services are those of "
            "another, causing damage to that other's goodwill."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Evidence Law — Admissibility and Burden of Proof",
        "text": (
            "The Indian Evidence Act, 1872 (IEA) is the primary statute governing admissibility "
            "of evidence in Indian courts, applicable to all judicial proceedings except affidavits "
            "and arbitral proceedings. Evidence must clear two hurdles to be considered by the "
            "court: relevance and admissibility. Relevance is governed by Sections 6 to 55 IEA "
            "and is a question of fact — evidence is relevant if it logically tends to prove or "
            "disprove a fact in issue. Admissibility is a question of law — relevant evidence "
            "may still be excluded if it is privileged, illegally obtained, or otherwise "
            "inadmissible under a specific rule. "
            "The burden of proof rests on the party who asserts a fact (Section 101 IEA) — "
            "'he who asserts must prove.' In civil cases, the standard of proof is the balance "
            "of probabilities — it is more likely than not that the fact is true. In criminal "
            "cases, the prosecution bears the burden of proving guilt beyond reasonable doubt — "
            "a high standard reflecting the principle that it is better for ten guilty persons "
            "to go free than for one innocent person to be convicted. The accused is presumed "
            "innocent until proven guilty. "
            "Hearsay evidence — an out-of-court statement tendered to prove the truth of its "
            "contents — is generally inadmissible under the IEA, subject to important exceptions: "
            "dying declarations (Section 32(1) — statements made by a person as to the cause of "
            "their death or circumstances of the transaction resulting in their death), res gestae "
            "(statements forming part of the same transaction), admissions and confessions, "
            "statements by persons who cannot be called as witnesses, and entries in books of "
            "account regularly kept. "
            "An admission by a party (Sections 17-23 IEA) is a statement, oral or documentary, "
            "that suggests any inference as to a fact in issue — it is admissible against the "
            "party who made it. A confession is an admission by an accused of guilt. Confessions "
            "made voluntarily before a magistrate under Section 164 CrPC are admissible. "
            "Confessions made to a police officer (Section 25 IEA) or while in police custody "
            "to any person (Section 26 IEA) are inadmissible — protecting accused persons from "
            "coerced confessions. Electronic records are admissible under Section 65B IEA, subject "
            "to a certificate of authenticity from the responsible person."
        ),
    },
]


def build_knowledge_base(embedder: SentenceTransformer):
    """Build ChromaDB in-memory collection from LEGAL_DOCUMENTS."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("legal_kb")

    texts = [doc["text"] for doc in LEGAL_DOCUMENTS]
    ids   = [doc["id"]   for doc in LEGAL_DOCUMENTS]
    metas = [{"topic": doc["topic"]} for doc in LEGAL_DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)
    print(f"✅  Knowledge base built — {len(LEGAL_DOCUMENTS)} documents loaded.")
    return collection


# ─────────────────────────────────────────────
# PART 2 — STATE DESIGN
# ─────────────────────────────────────────────

class CapstoneState(TypedDict):
    question:      str
    messages:      Annotated[list, operator.add]   # full conversation history
    route:         str                              # "retrieve" | "tool" | "memory_only"
    retrieved:     str                              # context chunks from ChromaDB
    sources:       list                             # topic names retrieved
    tool_result:   str                              # output of tool_node
    answer:        str                              # final answer from LLM
    faithfulness:  float                            # eval score 0.0–1.0
    eval_retries:  int                              # how many eval retries so far
    user_name:     str                              # extracted from conversation


# ─────────────────────────────────────────────
# PART 3 — NODE FUNCTIONS
# ─────────────────────────────────────────────

def make_nodes(llm, embedder, collection):
    """Factory: returns all node functions bound to shared resources."""

    # ── memory_node ──────────────────────────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        question = state["question"]

        # Sliding window — keep last 6 messages to avoid token overflow
        msgs = msgs[-6:] if len(msgs) > 6 else msgs
        msgs = msgs + [HumanMessage(content=question)]

        # Extract user name if introduced
        user_name = state.get("user_name", "")
        q_lower = question.lower()
        if "my name is" in q_lower:
            parts = q_lower.split("my name is")
            if len(parts) > 1:
                user_name = parts[1].strip().split()[0].capitalize()

        return {"messages": msgs, "user_name": user_name, "eval_retries": 0}

    # ── router_node ───────────────────────────────────────────────────────────
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        history  = state.get("messages", [])
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history[-4:]
        )

        prompt = f"""You are a routing assistant for a legal document assistant.
Given the user's question, decide the best route. Reply with EXACTLY one word — no punctuation, no explanation.

Routes:
- retrieve   → question asks about legal concepts, laws, rights, procedures, contracts, torts, evidence, IP (use the knowledge base)
- tool       → question asks about today's date, current time, or real-time web information
- memory_only → question is conversational (greetings, thanks, asking about the assistant, referring only to prior conversation)

Recent conversation:
{history_text}

User question: {question}

Route (one word only):"""

        response = llm.invoke([HumanMessage(content=prompt)])
        route = response.content.strip().lower().split()[0]
        if route not in {"retrieve", "tool", "memory_only"}:
            route = "retrieve"

        print(f"  [router] → {route}")
        return {"route": route}

    # ── retrieval_node ────────────────────────────────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        question  = state["question"]
        embedding = embedder.encode([question]).tolist()

        results = collection.query(
            query_embeddings=embedding,
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )

        docs    = results["documents"][0]
        metas   = results["metadatas"][0]
        sources = [m["topic"] for m in metas]

        context = "\n\n".join(
            f"[{meta['topic']}]\n{doc}"
            for doc, meta in zip(docs, metas)
        )

        print(f"  [retrieval] sources: {sources}")
        return {"retrieved": context, "sources": sources}

    # ── skip_node ─────────────────────────────────────────────────────────────
    def skip_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": [], "tool_result": ""}

    # ── tool_node ─────────────────────────────────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"].lower()
        result   = ""

        try:
            # Datetime tool
            if any(w in question for w in ["date", "day", "time", "today", "year", "month"]):
                now    = datetime.datetime.now()
                result = (
                    f"Current date and time: {now.strftime('%A, %d %B %Y, %I:%M %p')}. "
                    f"The year is {now.year}."
                )

            # Web search (DuckDuckGo — no API key needed)
            else:
                try:
                    from langchain_community.tools import DuckDuckGoSearchRun
                    search = DuckDuckGoSearchRun()
                    result = search.run(state["question"])
                except Exception as e:
                    result = f"Web search unavailable: {str(e)}. Please consult an up-to-date legal resource."

        except Exception as e:
            result = f"Tool error: {str(e)}"

        print(f"  [tool] result snippet: {result[:80]}...")
        return {"tool_result": result, "retrieved": "", "sources": []}

    # ── answer_node ───────────────────────────────────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        history      = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        user_name    = state.get("user_name", "")

        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history[-6:]
        )

        # Build context section
        context_section = ""
        if retrieved:
            context_section = f"\n\n=== LEGAL KNOWLEDGE BASE ===\n{retrieved}"
        if tool_result:
            context_section += f"\n\n=== TOOL RESULT ===\n{tool_result}"

        retry_instruction = ""
        if eval_retries > 0:
            retry_instruction = (
                "\n\nIMPORTANT: A previous answer scored low on faithfulness. "
                "Be more grounded in the context. Cite specific sections or principles."
            )

        name_line = f" The user's name is {user_name}." if user_name else ""

        system_prompt = f"""You are a knowledgeable and professional legal document assistant helping paralegals and junior lawyers.{name_line}

STRICT RULE: Answer ONLY from the provided legal knowledge base or tool results. 
If the answer is not in the context, say clearly: "I don't have information on that in my knowledge base. Please consult a qualified lawyer or authoritative legal source."
Never fabricate case citations, statutory provisions, or legal principles not present in the context.
Be precise, structured, and professional.{retry_instruction}"""

        full_prompt = f"""{system_prompt}
{context_section}

Conversation history:
{history_text}

User question: {question}

Answer:"""

        response = llm.invoke([HumanMessage(content=full_prompt)])
        answer   = response.content.strip()
        print(f"  [answer] length: {len(answer)} chars")
        return {"answer": answer}

    # ── eval_node ─────────────────────────────────────────────────────────────
    def eval_node(state: CapstoneState) -> dict:
        answer    = state.get("answer", "")
        retrieved = state.get("retrieved", "")
        retries   = state.get("eval_retries", 0)

        # Skip eval if no retrieved context (tool or memory_only routes)
        if not retrieved:
            print("  [eval] skipped (no retrieval context) → PASS")
            return {"faithfulness": 1.0, "eval_retries": retries}

        eval_prompt = f"""Rate how faithfully the answer is grounded in the provided context.
Score from 0.0 to 1.0:
  1.0 = every claim in the answer is directly supported by the context
  0.7 = most claims are supported; minor additions acceptable
  0.5 = roughly half the claims are supported
  0.0 = the answer introduces significant information not in the context

Context:
{retrieved[:1500]}

Answer:
{answer}

Reply with a single decimal number only (e.g., 0.85):"""

        try:
            response     = llm.invoke([HumanMessage(content=eval_prompt)])
            score_text   = response.content.strip().split()[0]
            faithfulness = float(score_text)
            faithfulness = max(0.0, min(1.0, faithfulness))
        except Exception:
            faithfulness = 0.8   # default pass if eval fails

        print(f"  [eval] faithfulness={faithfulness:.2f}, retries={retries}")
        return {"faithfulness": faithfulness, "eval_retries": retries + 1}

    # ── save_node ─────────────────────────────────────────────────────────────
    def save_node(state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        return {"messages": [AIMessage(content=answer)]}

    return (
        memory_node,
        router_node,
        retrieval_node,
        skip_node,
        tool_node,
        answer_node,
        eval_node,
        save_node,
    )


# ─────────────────────────────────────────────
# PART 4 — GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_graph(llm, embedder, collection):
    """Assemble and compile the LangGraph StateGraph."""
    (
        memory_node,
        router_node,
        retrieval_node,
        skip_node,
        tool_node,
        answer_node,
        eval_node,
        save_node,
    ) = make_nodes(llm, embedder, collection)

    # ── Routing functions ─────────────────────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":
            return "tool"
        if r == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score < FAITHFULNESS_THRESHOLD and retries <= MAX_EVAL_RETRIES:
            print(f"  [eval_decision] RETRY (score={score:.2f})")
            return "answer"       # retry the answer node
        print(f"  [eval_decision] PASS → save")
        return "save"

    # ── Build graph ───────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",    memory_node)
    graph.add_node("router",    router_node)
    graph.add_node("retrieve",  retrieval_node)
    graph.add_node("skip",      skip_node)
    graph.add_node("tool",      tool_node)
    graph.add_node("answer",    answer_node)
    graph.add_node("eval",      eval_node)
    graph.add_node("save",      save_node)

    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    # Conditional edges
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    app = graph.compile(checkpointer=MemorySaver())
    print("✅  Graph compiled successfully.")
    return app


# ─────────────────────────────────────────────
# INITIALISATION (called once by Streamlit)
# ─────────────────────────────────────────────

def initialise():
    """Load all heavy resources. Returns (app, embedder, collection)."""
    print("Loading LLM …")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=0.1,
    )

    print("Loading sentence embedder …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Building knowledge base …")
    collection = build_knowledge_base(embedder)

    print("Assembling graph …")
    app = build_graph(llm, embedder, collection)

    return app, embedder, collection


# ─────────────────────────────────────────────
# PART 5 — ASK HELPER
# ─────────────────────────────────────────────

def ask(app, question: str, thread_id: str = "default") -> dict:
    """Send a question to the agent and return the result state."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: CapstoneState = {
        "question":     question,
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 1.0,
        "eval_retries": 0,
        "user_name":    "",
    }
    result = app.invoke(initial_state, config=config)
    return result


# ─────────────────────────────────────────────
# QUICK SMOKE-TEST (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app, embedder, collection = initialise()

    test_questions = [
        "What are the essential elements of a valid contract?",
        "What happens if someone breaches a contract?",
        "What rights does an arrested person have?",
        "What is the burden of proof in criminal cases?",
        "What is the difference between copyright and trademark?",
        "What is vicarious liability?",
        "What is today's date?",
        "Hello, my name is Priya. How can you help me?",
        "What did I just tell you my name was?",              # memory test
        "What is the cure rate for pancreatic cancer?",       # out-of-scope test
    ]

    thread = "smoke_test"
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(app, q, thread_id=thread)
        print(f"A: {result['answer'][:300]}")
        print(f"   route={result['route']} | faithfulness={result['faithfulness']:.2f} | sources={result['sources']}")
