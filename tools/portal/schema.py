"""Supabase schema constants for the Fullhouse portal."""

SUPABASE_URL = "https://zqarejzswyaxsieulhps.supabase.co"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpxYXJlanpzd3lheHNpZXVsaHBzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUzODczMjMsImV4cCI6MjA5MDk2MzMyM30."
    "sGkrhovXW7l3TClRM7rjkcXhITNFDGjILiDB99uL5rc"
)

TABLES = {
    "tournaments": {
        "select": "id,name,phase,current_round,total_rounds,n_finalists,starts_at,created_at",
        "order": "starts_at.asc",
    },
    "matches": {
        "select": "id,tournament_id,round,table_index,status,n_hands,started_at,completed_at,error_message",
        "order": "round.asc,table_index.asc",
    },
    "match_bots": {
        "select": "match_id,bot_id,seat,final_stack,chip_delta",
        "order": "match_id.asc,seat.asc",
    },
    "hands": {
        "select": "id,match_id,hand_num,street,pot,community_cards,action_log,revealed_cards,played_at",
        "order": "match_id.asc,hand_num.asc",
    },
    "hand_winners": {
        "select": "hand_id,bot_id,amount",
        "order": "hand_id.asc",
    },
    "bots": {
        "select": "id,user_id,bot_name,version,status,error_message,submitted_at",
        "order": "submitted_at.asc",
    },
    "leaderboard": {
        "select": "tournament_id,bot_id,rank,cumulative_delta,matches_played,updated_at",
        "order": "rank.asc",
    },
}

DEFAULT_FULL_TABLES = (
    "tournaments",
    "matches",
    "match_bots",
    "hands",
    "hand_winners",
    "bots",
    "leaderboard",
)
METADATA_TABLES = ("tournaments", "matches", "match_bots", "bots", "leaderboard")
HANDS_SELECT = (
    "id,match_id,hand_num,street,pot,community_cards,action_log,"
    "revealed_cards,played_at,hand_winners(bot_id,amount)"
)

DEFAULT_LEADERBOARD_NAME = "Fullhouse 2026 Qualifier"
DEFAULT_SOURCE_NAME_CONTAINS = "Qualifier"
