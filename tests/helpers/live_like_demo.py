from apps.api.app.schemas import RawRetrievalResponse, RetrievalCandidate


LIVE_LIKE_DEMO_QUERY = (
    "Poate angajatorul sa-mi scada salariul fara act aditional?"
)


LIVE_LIKE_UNITS = [
    {
        "id": "ro.codul_muncii.art_16.alin_1",
        "article_number": "16",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Contractul individual de muncă se încheie în baza "
            "consimțământului părților, în forma scrisă, în limba română, "
            "anterior începerii activității. Obligația de încheiere a "
            "contractului individual de muncă în forma scrisă revine "
            "angajatorului."
        ),
    },
    {
        "id": "ro.codul_muncii.art_196.alin_2",
        "article_number": "196",
        "paragraph_number": "2",
        "letter_number": None,
        "raw_text": (
            "Modalitatea concretă de formare profesională, drepturile și "
            "obligațiile părților, durata formării profesionale, precum și "
            "orice alte aspecte legate de formarea profesională fac obiectul "
            "unor acte adiționale la contractele individuale de muncă."
        ),
    },
    {
        "id": "ro.codul_muncii.art_42.alin_1",
        "article_number": "42",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Locul muncii poate fi modificat unilateral de către angajator "
            "prin delegarea sau detașarea salariatului într-un alt loc de "
            "muncă decât cel prevăzut în contractul individual de muncă."
        ),
    },
    {
        "id": "ro.codul_muncii.art_254.alin_3",
        "article_number": "254",
        "paragraph_number": "3",
        "letter_number": None,
        "raw_text": (
            "În situația în care angajatorul constată că salariatul său a "
            "provocat o pagubă din vina și în legătură cu munca sa, va putea "
            "solicita salariatului, printr-o notă de constatare și evaluare "
            "a pagubei, recuperarea contravalorii acesteia, prin acordul "
            "părților."
        ),
    },
    {
        "id": "ro.codul_muncii.art_41.alin_1",
        "article_number": "41",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Contractul individual de muncă poate fi modificat numai prin "
            "acordul părților."
        ),
    },
    {
        "id": "ro.codul_muncii.art_41.alin_3",
        "article_number": "41",
        "paragraph_number": "3",
        "letter_number": None,
        "raw_text": (
            "Modificarea contractului individual de muncă se referă la "
            "oricare dintre următoarele elemente: durata contractului, locul "
            "muncii, felul muncii, condițiile de muncă, salariul, timpul de "
            "muncă și timpul de odihnă."
        ),
    },
    {
        "id": "ro.codul_muncii.art_41.alin_3.lit_e",
        "article_number": "41",
        "paragraph_number": "3",
        "letter_number": "e",
        "parent_id": "ro.codul_muncii.art_41.alin_3",
        "raw_text": "e) salariul;",
    },
    {
        "id": "ro.codul_muncii.art_17.alin_3.lit_b",
        "article_number": "17",
        "paragraph_number": "3",
        "letter_number": "b",
        "parent_id": "ro.codul_muncii.art_17.alin_3",
        "raw_text": (
            "b) locul muncii sau, in lipsa unui loc de munca fix, "
            "posibilitatea ca salariatul sa munceasca in diverse locuri;"
        ),
    },
    {
        "id": "ro.codul_muncii.art_35.alin_1",
        "article_number": "35",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Orice salariat are dreptul de a munci la angajatori diferiti "
            "sau la acelasi angajator, in baza unor contracte individuale de munca."
        ),
    },
    {
        "id": "ro.codul_muncii.art_166",
        "article_number": "166",
        "paragraph_number": None,
        "letter_number": None,
        "raw_text": "Salariul se plateste in bani cel putin o data pe luna.",
    },
    {
        "id": "ro.codul_muncii.art_260.alin_1.lit_a",
        "article_number": "260",
        "paragraph_number": "1",
        "letter_number": "a",
        "parent_id": "ro.codul_muncii.art_260.alin_1",
        "raw_text": (
            "a) primirea la munca a uneia sau a mai multor persoane fara "
            "incheierea unui contract individual de munca constituie contraventie."
        ),
    },
]


def live_like_retrieval_candidates() -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for index, unit in enumerate(LIVE_LIKE_UNITS, start=1):
        retrieval_score = round(0.96 - index * 0.02, 6)
        candidates.append(
            RetrievalCandidate(
                unit_id=unit["id"],
                rank=index,
                retrieval_score=retrieval_score,
                score_breakdown={
                    "bm25": retrieval_score,
                    "dense": 0.0,
                    "domain_match": 1.0,
                },
                why_retrieved="live_like_regression_fixture",
                unit={
                    **unit,
                    "law_id": "ro.codul_muncii",
                    "law_title": "Codul muncii",
                    "status": "active",
                    "legal_domain": "munca",
                    "source_url": "https://legislatie.just.ro/test",
                    "type": "alineat",
                    "normalized_text": unit["raw_text"].casefold(),
                    "hierarchy_path": [
                        "Codul muncii",
                        f"Art. {unit['article_number']}",
                        f"Alin. ({unit['paragraph_number']})",
                    ],
                },
            )
        )
    return candidates


class LiveLikeRawRetriever:
    async def retrieve(self, plan, *, top_k: int = 50, debug: bool = False):
        candidates = live_like_retrieval_candidates()[:top_k]
        return RawRetrievalResponse(
            candidates=candidates,
            retrieval_methods=["live_like_regression_fixture"],
            warnings=[],
            debug={"candidate_count": len(candidates)} if debug else None,
        )
