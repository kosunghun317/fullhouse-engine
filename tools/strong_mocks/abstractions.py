"""State abstractions for benchmark-only strong mock training."""

from __future__ import annotations

import eval7

from tools.strong_mocks.features import RANK_ORDER, bucket_id, extract_features, rank_value


def _rank_index(rank: str) -> int:
    return RANK_ORDER.index(rank)


def _preflop_class_index(cards: list[str]) -> int:
    if len(cards) < 2:
        return 0
    c1, c2 = cards[:2]
    r1, r2 = c1[0], c2[0]
    i1, i2 = _rank_index(r1), _rank_index(r2)
    high, low = max(i1, i2), min(i1, i2)
    if high == low:
        return high
    suited = len(c1) > 1 and len(c2) > 1 and c1[1] == c2[1]
    # 13 pair slots, then upper-triangular suited/offsuit non-pair classes.
    offset = 13
    combo = high * (high - 1) // 2 + low
    return offset + combo * 2 + (0 if suited else 1)


def _made_rank(cards: list[str]) -> int:
    if len(cards) < 5:
        return 0
    try:
        score = eval7.evaluate([eval7.Card(card) for card in cards])
        hand_type = str(eval7.handtype(score)).lower()
    except Exception:
        return 0
    if "straight flush" in hand_type:
        return 8
    if "four" in hand_type:
        return 7
    if "full house" in hand_type:
        return 6
    if "flush" in hand_type:
        return 5
    if "straight" in hand_type:
        return 4
    if "three" in hand_type:
        return 3
    if "two pair" in hand_type:
        return 2
    if "pair" in hand_type:
        return 1
    return 0


def _draw_flags(cards: list[str], board: list[str]) -> tuple[int, int]:
    if len(board) >= 5:
        return 0, 0
    ranks = {rank_value(card) for card in cards}
    if 14 in ranks:
        ranks.add(1)
    straight = 0
    for start in range(1, 11):
        if len({start, start + 1, start + 2, start + 3, start + 4} & ranks) >= 4:
            straight = 1
            break
    suits = {}
    for card in cards:
        if len(card) > 1:
            suits[card[1]] = suits.get(card[1], 0) + 1
    flush = int(max(suits.values(), default=0) >= 4)
    return straight, flush


def _board_bucket(board: list[str]) -> int:
    if len(board) < 3:
        return 0
    ranks = sorted((rank_value(card) for card in board), reverse=True)
    high = min(4, max(0, (ranks[0] - 2) // 3))
    broadways = min(3, sum(1 for rank in ranks if rank >= 10))
    paired = min(2, len(ranks) - len(set(ranks)))
    suits = {}
    for card in board:
        if len(card) > 1:
            suits[card[1]] = suits.get(card[1], 0) + 1
    suit_bucket = min(3, max(suits.values(), default=0))
    unique = sorted(set(ranks))
    connected = min(3, sum(1 for i in range(1, len(unique)) if unique[i] - unique[i - 1] <= 2))
    return (((high * 4 + broadways) * 3 + paired) * 4 + suit_bucket) * 4 + connected


def _stable_mix(values: list[int], bucket_count: int) -> int:
    acc = 2166136261
    for value in values:
        acc ^= int(value) & 0xFFFFFFFF
        acc = (acc * 16777619) & 0xFFFFFFFF
    return int(acc % max(1, bucket_count))


def cfr_pokerbot_bucket_id(state: dict, bucket_count: int = 4096) -> int:
    """Fullhouse-compatible abstraction inspired by CFR_pokerbot.

    CFR_pokerbot targets MIT Toss Hold'em, which has three hole cards and a
    face-up discard. Fullhouse is normal two-card NLHE, so this abstraction
    keeps the compatible ideas only: explicit preflop class, board structure,
    made-hand/draw class, pot/owed bucket, position, active players, and action
    pressure.
    """
    features = extract_features(state)
    street = int(features[1] * 0 + features[2] * 1 + features[3] * 2 + features[4] * 3)
    hole = _preflop_class_index(list(state.get("your_cards", [])))
    board = list(state.get("community_cards", []))
    all_cards = list(state.get("your_cards", [])) + board
    made = _made_rank(all_cards)
    straight_draw, flush_draw = _draw_flags(all_cards, board)
    hand_bucket = hole if street == 0 else 169 + made * 4 + straight_draw * 2 + flush_draw
    board_bucket = _board_bucket(board)
    pot_bucket = min(7, int(features[6] * 8))
    owed_bucket = min(5, int(features[5] * 6))
    stack_bucket = min(5, int(features[7] * 6))
    active_bucket = min(5, int(features[10] * 6))
    position_bucket = min(5, int(features[11] * 6))
    pressure_bucket = min(5, int(features[29] * 4 + features[30] * 2))
    return _stable_mix([
        street,
        hand_bucket,
        board_bucket,
        pot_bucket,
        owed_bucket,
        stack_bucket,
        active_bucket,
        position_bucket,
        pressure_bucket,
    ], bucket_count)


def abstract_bucket_id(state: dict, bucket_count: int = 4096, abstraction: str = "feature") -> int:
    if abstraction == "cfr-pokerbot":
        return cfr_pokerbot_bucket_id(state, bucket_count)
    return bucket_id(state, bucket_count)
