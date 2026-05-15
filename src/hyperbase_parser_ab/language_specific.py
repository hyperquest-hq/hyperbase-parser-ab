from spacy.tokens import Token


def apply_candidate_overrides(
    language: str,
    token: Token,
    candidates: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], bool]:
    # Per-language overrides applied to the atomizer's top-K candidate
    # list for a single token. Returns the (possibly-reordered)
    # candidates and a force_uncertain flag: when True, the caller must
    # treat this token as uncertain regardless of probability gates so
    # the demoted candidate is offered as an alternative in the
    # hill-climb.
    if language == "pt":
        return _apply_pt_overrides(token, candidates)
    return candidates, False


def _apply_pt_overrides(
    token: Token,
    candidates: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], bool]:
    # Portuguese: the preposition "de" is frequently classified as a
    # modifier (M*) when it should be a builder (B*). When B* is the
    # runner-up, promote it to top-1 and mark the token as uncertain so
    # the original M* candidate is retried on the next assignment.
    if len(candidates) < 2:
        return candidates, False
    if token.text.lower() != "de":
        return candidates, False
    top1_label = candidates[0][0]
    top2_label = candidates[1][0]
    top1_prob = candidates[0][1]
    if not top1_label or not top2_label:
        return candidates, False
    if top1_label[:1] == "M" and top2_label[:1] == "B" and top1_prob < 0.8:
        reordered = list(candidates)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        return reordered, True
    return candidates, False
