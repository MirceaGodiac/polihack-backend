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
            "Contractul individual de munca se incheie in baza "
            "consimtamantului partilor, in forma scrisa, in limba romana, "
            "anterior inceperii activitatii. Obligatia de incheiere a "
            "contractului individual de munca in forma scrisa revine "
            "angajatorului."
        ),
    },
    {
        "id": "ro.codul_muncii.art_196.alin_2",
        "article_number": "196",
        "paragraph_number": "2",
        "letter_number": None,
        "raw_text": (
            "Modalitatea concreta de formare profesionala, drepturile si "
            "obligatiile partilor, durata formarii profesionale, precum si "
            "orice alte aspecte legate de formarea profesionala fac obiectul "
            "unor acte aditionale la contractele individuale de munca."
        ),
    },
    {
        "id": "ro.codul_muncii.art_42.alin_1",
        "article_number": "42",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Locul muncii poate fi modificat unilateral de catre angajator "
            "prin delegarea sau detasarea salariatului intr-un alt loc de "
            "munca decat cel prevazut in contractul individual de munca."
        ),
    },
    {
        "id": "ro.codul_muncii.art_254.alin_3",
        "article_number": "254",
        "paragraph_number": "3",
        "letter_number": None,
        "raw_text": (
            "In situatia in care angajatorul constata ca salariatul sau a "
            "provocat o paguba din vina si in legatura cu munca sa, va putea "
            "solicita salariatului, printr-o nota de constatare si evaluare "
            "a pagubei, recuperarea contravalorii acesteia, prin acordul "
            "partilor."
        ),
    },
    {
        "id": "ro.codul_muncii.art_41.alin_1",
        "article_number": "41",
        "paragraph_number": "1",
        "letter_number": None,
        "raw_text": (
            "Contractul individual de munca poate fi modificat numai prin "
            "acordul partilor."
        ),
    },
    {
        "id": "ro.codul_muncii.art_41.alin_3",
        "article_number": "41",
        "paragraph_number": "3",
        "letter_number": None,
        "raw_text": (
            "Modificarea contractului individual de munca se refera la "
            "oricare dintre urmatoarele elemente: durata contractului, locul "
            "muncii, felul muncii, conditiile de munca, salariul, timpul de "
            "munca si timpul de odihna."
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
