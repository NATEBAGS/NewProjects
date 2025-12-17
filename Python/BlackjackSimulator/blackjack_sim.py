import random
import math
import matplotlib.pyplot as plt
import statistics

# Turn on for hand-by-hand printing
VERBOSE = True
PLOT_RESULTS = True
# General Blackjack rules (change as needed)
DEALER_HITS_SOFT_17 = False
ALLOW_DOUBLE_AFTER_SPLIT = False
# This stores the card counting strategy values per card A, 2, 3, ..., K
STRATEGY = (0, 1, 1, 2, 2, 1, 1, 0, 0, -2, -2, -2, -2)
# This stores a insurance strategy of you choice
INSURANCE_STRAT = (1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2)

UPCARD_MAPPING = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7, '10': 8,
    'J': 8, 'Q': 8, 'K': 8, 'A': 9
}

# Maps the card to the count value for the counting system
COUNT_MAP = {
    'A': STRATEGY[0], '2': STRATEGY[1], '3': STRATEGY[2], '4': STRATEGY[3], '5': STRATEGY[4],
    '6': STRATEGY[5], '7': STRATEGY[6], '8': STRATEGY[7], '9': STRATEGY[8],
    '10': STRATEGY[9], 'J': STRATEGY[10], 'Q': STRATEGY[11], 'K': STRATEGY[12],
}
# Maps the card to the count value for the insurance
INSURANCE_COUNT_MAP = {
    'A': INSURANCE_STRAT[0], '2': INSURANCE_STRAT[1], '3': INSURANCE_STRAT[2], '4': INSURANCE_STRAT[3], '5': INSURANCE_STRAT[4],
    '6': INSURANCE_STRAT[5], '7': INSURANCE_STRAT[6], '8': INSURANCE_STRAT[7], '9': INSURANCE_STRAT[8],
    '10': INSURANCE_STRAT[9], 'J': INSURANCE_STRAT[10], 'Q': INSURANCE_STRAT[11], 'K': INSURANCE_STRAT[12],
}

# Splitting charts

""" 
HOW TO READ THE CHARTS:
Starting from the dealers upcard of 2 through A, what should the player do
Y stands for Yes (split) N stands for no, number values stand for verified HIOPT2 deviations
If a true count goes over the specified number, then do the deviation. + stands for that number and greater
- stands for that number and less 
"""
# HIOPT2 splitting charts
HIOPT2_SPLITTING = {
    'A': ['Y'] * 9 + ['N'],
    '10': ['N'] * 2 + ['10+', '8+', '7+'] + ['N'] * 5,
    '9': ['Y'] * 5 + ['14+', 'Y', '-25+', 'N', '11+'],
    '8': ['Y'] * 7 + ['31-'] + ['N'] * 2,
    '7': ['Y'] * 6 + ['N'] * 4,
    '6': ['4+'] + ['-1+', '-4+', '-8+', '-11+'] + ['N'] * 5,
    '5': ['N'] * 10,
    '4': ['N'] * 10,
    '3': ['13+', '6+', '0+'] + ['Y'] * 3 + ['N'] * 4,
    '2': ['12+', '4+'] + ['Y'] * 4 + ['N'] * 4,
}
# HIOPT2 soft Hit/Stand/DD charts. Number deviations are only for double downs
HIOPT2_SOFT = {
    20: ['18+'] + ['15+', '12+', '10+', '9+'] + ['S'] * 5,
    19: ['14+', '8+', '6+', '3+'] + ['D'] + ['27+'] + ['S'] * 4,
    18: ['2+'] + ['D'] * 4 + ['28+'] + ['S'] + ['H'] * 3,
    17: ['2+'] + ['D'] * 4 + ['22+'] + ['H'] * 4,
    16: ['19+'] + ['6+'] + ['D'] * 3 + ['H'] * 5,
    15: ['20+'] + ['8+'] + ['D'] * 3 + ['H'] * 5,
    14: ['18+'] + ['9+'] + ['2+'] + ['D'] * 2 + ['H'] * 5,
    13: ['18+'] + ['10+'] + ['5+'] + ['D'] * 2 + ['H'] * 5,
}
# HIOPT2 Hard Hitting
HIOPT2_HARD = {
    17: ['S'] * 10,
    16: ['S'] * 5 + ['14-', '13-', '7-'] + ['S'] + ['13-'],
    15: ['S'] * 5 + ['H'] * 2 + ['14-', '6-'] + ['H'],
    14: ['S'] * 5 + ['H'] * 3 + ['14-'] + ['H'],
    13: ['S'] * 5 + ['H'] * 5,
    12: ['5-'] + ['2-'] + ['0-'] + ['-2-'] * 2 + ['H'] * 5,
}
# HIOPT2 Hard double down
HIOPT2_HARDDUB = {
    11: ['D'] * 8 + ['H'] * 2,
    10: ['D'] * 8 + ['H'] * 2,
    9: ['3+'] + ['D'] * 4 + ['6+', '15+'] + ['H'] * 3,
    8: ['H'] * 2 + ['11+', '6+', '4+'] + ['H'] * 5,
}

# Regular Strategy Charts for European S17, no DAS (No deviations, basic strategy player)

# Splitting strategy
SPLITTING = {
    'A': ['Y'] * 9 + ['N'],
    '10': ['N'] * 10,
    '9': ['Y'] * 5 + ['N'] + ['Y'] * 2 + ['N'] * 2,
    '8': ['Y'] * 8 + ['N'] * 2,
    '7': ['Y'] * 6 + ['N'] * 4,
    '6': ['Y'] * 5 + ['N'] * 5,
    '5': ['N'] * 10,
    '4': ['N'] * 3 + ['Y'] * 2 + ['N'] * 5,
    '3': ['Y'] * 6 + ['N'] * 4,
    '2': ['Y'] * 6 + ['N'] * 4,
}
# Soft hand strategy
SOFT = {
    21: ['S'] * 10,
    20: ['S'] * 10,
    19: ['S'] * 10,
    18: ['S'] * 7 + ['H'] * 3,
    17: ['H'] * 10,
    16: ['H'] * 10,
    15: ['H'] * 10,
    14: ['H'] * 10,
    13: ['H'] * 10,
}
# Hard hand strategy
HARD = {
    17: ['S'] * 10,
    16: ['S'] * 5 + ['H'] * 5,
    15: ['S'] * 5 + ['H'] * 5,
    14: ['S'] * 5 + ['H'] * 5,
    13: ['S'] * 5 + ['H'] * 5,
    12: ['H'] * 2 + ['S'] * 3 + ['H'] * 5,
}
# Hard double down strategy
HARDDUB = {
    11: ['D'] * 8 + ['H'] * 2,
    10: ['D'] * 8 + ['H'] * 2,
    9: ['H'] + ['D'] * 4 + ['H'] * 5,
    8: ['H'] * 10,
}

# Turn on to print rounds
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Handles identifying each card
class Card:
    def __init__(self, rank: str):
        self.rank = rank

    def __str__(self):
        return self.rank

# Handles the deck instantiation
class Deck:
    def __init__(self, num_decks: int):
        self.num_decks = num_decks
        self.cards = []
        self.cut_card_position = None

        # running counts on seen cards only
        self.count = 0
        self.true_count = 0
        self.betting_count = 0
        self.insurance_count = 0

        # for betting count
        self.aces_remaining = 0
        # Make sure every shoe resets
        self.initialize_deck()

    def initialize_deck(self):
        """ Sets up the blackjack shoe """
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(rank) for rank in ranks for _ in range(4 * self.num_decks)]
        self.reshuffle()
        self.set_cut_card_position()

    def reshuffle(self):
        """ Shuffles the deck and resets counts """
        random.shuffle(self.cards)
        self.count = 0
        self.true_count = 0
        self.betting_count = 0
        self.insurance_count = 0
        self.aces_remaining = 4 * self.num_decks

    def set_cut_card_position(self):
        """ Handles the placing of the cutcard """
        # The cut card is placed randomly for half penetration in this use case
        burn = random.randint(190, 235)
        burn = min(burn, len(self.cards) - 1)
        self.cut_card_position = len(self.cards) - burn

    def update_count_seen(self, card: Card):
        """ Updates the counts """
        self.count += COUNT_MAP[card.rank]
        self.insurance_count += INSURANCE_COUNT_MAP[card.rank]

    def deal_card(self, seen: bool = True) -> Card:
        """ Handles the dealing of the cards"""
        if not self.cards:
            self.initialize_deck()
        # Take the card off the top of the deck
        card = self.cards.pop()

        # Track remaining aces for betting count
        if card.rank == 'A':
            self.aces_remaining -= 1

        # Only update running/insurance count if card is seen
        if seen:
            self.update_count_seen(card)

        return card

    def cards_left(self) -> int:
        """ Returns how many cards are left """
        return len(self.cards)

    def get_true_count(self) -> float:
        """ Gets the true count of the current shoe """
        remaining_decks = max(len(self.cards) / 52.0, 1e-9)
        self.true_count = self.count / remaining_decks
        return self.true_count

    def get_betting_count(self) -> float:
        """ Calculates the betting count which modifies the true count in accordance to the number of aces left """
        # We expect to see an ace every 13 cards
        expected_aces = len(self.cards) / 13.0
        # Use this to find how many extra aces we have in the shoe
        ace_surplus = self.aces_remaining - expected_aces
        remaining_decks = max(len(self.cards) / 52.0, 1e-9)

        # Betting Count = Running count + 2 * (ace surplus)
        self.betting_count = (self.count + (ace_surplus * 2.0)) / remaining_decks
        return self.betting_count

    def get_insurance_strat(self) -> bool:
        """ Gets the insurance count to see if we need to buy insurance"""
        return (self.insurance_count - (self.num_decks * 4)) > 0


class Player:
    """ Handles how the players will be created """
    def __init__(self, name, game, initial_balance=1000):
        self.name = name
        self.game = game

        self.hands = [[]]
        # Used if split happens
        self.hand_bets = [0.0]

        # Setting up the tracking for the data
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.blackjacks = 0
        self.balance = float(initial_balance)
        self.bet = 0.0
        self.insurance_bet = 0.0

    def reset_hands(self):
        """ Resets the hands and bets on the table"""
        self.hands = [[]]
        self.hand_bets = [0.0]
        self.bet = 0.0
        self.insurance_bet = 0.0

    def place_bet(self, amount: float):
        """ Handles the placing of a bet logic """
        amount = float(amount)
        if amount <= 0:
            self.bet = 0.0
            self.hand_bets[0] = 0.0
            return

        if amount <= self.balance:
            self.bet = amount
            self.hand_bets[0] = amount
            self.balance -= amount
            log(f"{self.name} placed bet ${amount:.2f}. Bal=${self.balance:.2f}")
        else:
            log(f"{self.name} cannot afford bet ${amount:.2f}. Bal=${self.balance:.2f}")
            self.bet = 0.0
            self.hand_bets[0] = 0.0

    def can_cover_extra_bet(self, amount: float) -> bool:
        """ Checks if a player can double/split/insurance """
        return amount <= self.balance

    def double_down(self, hand_index: int) -> bool:
        """ Handles the double down action for a player """
        bet = self.hand_bets[hand_index]
        if bet <= self.balance:
            self.balance -= bet
            self.hand_bets[hand_index] = bet * 2.0
            return True
        return False

    def split_hand(self, hand_index: int, new_card_1: Card, new_card_2: Card) -> bool:
        """ Handles the split hand for the player. No resplitting allowed in the current rules """
        # need to place an additional bet equal to the original bet
        bet = self.hand_bets[hand_index]
        if bet > self.balance:
            return False

        self.balance -= bet
        self.hand_bets.append(bet)

        c1 = self.hands[hand_index][0]
        c2 = self.hands[hand_index][1]

        self.hands[hand_index] = [c1, new_card_1]
        self.hands.append([c2, new_card_2])
        return True

    def add_card_to_hand(self, card: Card, hand_index=0):
        """ Add a card to a players hand """
        self.hands[hand_index].append(card)

    def hand_total(self, hand_index=0):
        """ Sums the total of the hand, including Ace functionality """
        return Counter.sumHand([c.rank for c in self.hands[hand_index]])

    def has_blackjack(self, hand_index=0) -> bool:
        """ Checks if a certain player/dealer has blackjack """
        ranks = [c.rank for c in self.hands[hand_index]]
        return (len(ranks) == 2) and ('A' in ranks) and any(r in ('10', 'J', 'Q', 'K') for r in ranks)

    def is_busted(self, hand_index=0) -> bool:
        """ Boolean function to see if a player/dealer busted """
        total, _ = self.hand_total(hand_index)
        return total > 21

    def show_hand(self, hand_index=0):
        """ Shows the value of the hand """
        total, soft = self.hand_total(hand_index)
        s = "Soft" if soft else "Hard"
        desc = " ".join(str(c) for c in self.hands[hand_index])
        log(f"{self.name} ({s} {total}): {desc}")

    def basic_best_move(self, upcard, hand_index=0):
        """ Gets the best move for european blackjack basic strategy """
        hand = self.hands[hand_index]
        i = UPCARD_MAPPING[upcard.upper()]
        hand_sum, soft = self.hand_total(hand_index)

        if not soft and hand_sum in HARDDUB:
            play = HARDDUB[hand_sum][i]
            return 'DOUBLE' if (play == 'D' and len(hand) == 2) else 'HIT'

        if len(hand) == 2 and hand[0].rank == hand[1].rank:
            card = hand[0].rank
            if card in ('K', 'J', 'Q'):
                card = '10'
            if SPLITTING.get(card, ['N'] * 10)[i] == 'Y':
                return 'SPLIT'

        if soft and hand_sum in SOFT:
            play = SOFT[hand_sum][i]
            if play == 'S':
                return 'STAND'
            if play == 'D':
                return 'DOUBLE' if len(hand) == 2 else 'HIT'
            return 'HIT'

        if (not soft) and (hand_sum in HARD):
            play = HARD[hand_sum][i]
            if play == 'D':
                return 'DOUBLE' if len(hand) == 2 else 'HIT'
            return 'HIT' if play == 'H' else 'STAND'

        return 'STAND' if (not soft and hand_sum > 15) else 'HIT'

    def bestMove(self, upcard: str, hand_index: int) -> str:
        """ This calculates the best move for a card counter that uses a deviation-based startegy (HIOPT2) """
        def isValidTC(tc: str) -> bool:
            """ This checks if a hand has a valid deviation """
            try:
                n, sym = tc[:-1], tc[-1]
                tc_now = self.game.deck.get_true_count()
                if sym == '+':
                    return tc_now >= float(n)
                if sym == '-':
                    return tc_now <= float(n)
                return False
            except Exception:
                return False

        i = UPCARD_MAPPING[upcard.upper()]
        ranks = [c.rank for c in self.hands[hand_index]]
        handSum, soft = Counter.sumHand(ranks)

        if soft and handSum == 21:
            return 'STAND'

        if (not soft) and (handSum in HIOPT2_HARDDUB):
            play = HIOPT2_HARDDUB[handSum][i]
            if (play == 'D' or isValidTC(play)) and len(ranks) == 2:
                return 'DOUBLE'
            return 'HIT'

        if len(ranks) == 2 and ranks[0] == ranks[1]:
            card = ranks[0].upper()
            if card in ('K', 'J', 'Q'):
                card = '10'
            play = HIOPT2_SPLITTING[card][i]
            if play == 'Y' or isValidTC(play):
                return 'SPLIT'

        if soft and handSum in HIOPT2_SOFT:
            play = HIOPT2_SOFT[handSum][i]
            if play == 'S':
                return 'STAND'
            if play == 'H':
                return 'HIT'
            if play == 'D' or isValidTC(play):
                return 'DOUBLE' if len(ranks) == 2 else 'HIT'
            return 'STAND' if handSum >= 19 else 'HIT'

        if (not soft) and (handSum in HIOPT2_HARD):
            play = HIOPT2_HARD[handSum][i]
            if play == 'D':
                return 'DOUBLE' if len(ranks) == 2 else 'HIT'
            if play == 'H' or isValidTC(play):
                return 'HIT'
            return 'STAND'

        return 'STAND' if (not soft and handSum > 15) else 'HIT'


class BlackjackGame:
    """ This class handles the normal operations of a EURO blackjack game (no hole card peeking) """
    def __init__(self, num_decks=1, num_players=1):
        self.deck = Deck(num_decks)
        self.players = [Player(f"Player {i+1}", self, initial_balance=1000) for i in range(num_players)]
        self.dealer = Player("Dealer", self, initial_balance=0)

        self.round_results = []
        self.num_rounds = 0

        self.hands_played = [0] * num_players
        self._dealer_hole_revealed = False

    def reset_hands(self):
        """ Reset the hands of the current round """
        for p in self.players:
            p.reset_hands()
        self.dealer.reset_hands()
        self._dealer_hole_revealed = False

    def reveal_dealer_hole_card(self):
        """ Dealer flips their undercard """
        if self._dealer_hole_revealed:
            return
        if self.dealer.hands and self.dealer.hands[0] and len(self.dealer.hands[0]) >= 2:
            hole = self.dealer.hands[0][1]
            self.deck.update_count_seen(hole)  # now it's visible
        self._dealer_hole_revealed = True

    def collect_bets(self):
        """ Collect (place) the bets on the table """
        bc = self.deck.get_betting_count()
        for p in self.players:
            if p.name == "Player 1":
                if bc < 0.5:
                    p.place_bet(0)
                    continue
                bal = p.balance
                base = 5
                kelly_bet = round(((bc - 0.5) / 200.0) * 0.77 * bal)
                bet_amt = max(base, kelly_bet)
                p.place_bet(bet_amt)
            else:
                p.place_bet(5)

    def deal_initial_cards(self):
        """ Everyone gets 2 cards to begin with """
        # deal 2 cards to active bettors only
        for _ in range(2):
            for p in self.players:
                if p.bet > 0:
                    p.add_card_to_hand(self.deck.deal_card(seen=True))
            if _ == 0:
                # dealer upcard is seen
                self.dealer.add_card_to_hand(self.deck.deal_card(seen=True))
            else:
                # dealer hole card is NOT seen until reveal
                self.dealer.add_card_to_hand(self.deck.deal_card(seen=False))

        for p in self.players:
            if p.bet > 0:
                p.show_hand()

        log("Dealer upcard:", self.dealer.hands[0][0])

        if self.dealer.hands[0][0].rank == 'A':
            self.offer_insurance()

    def offer_insurance(self):
        """ Asks the players if they would like to take insurance"""
        # only Player 1 in your original logic
        for p in self.players:
            # Basic strategy says to never buy insurance
            if p.name != "Player 1" or p.bet <= 0:
                continue

            if self.deck.get_insurance_strat():
                insurance_bet = p.bet / 2.0
                if insurance_bet <= p.balance:
                    p.balance -= insurance_bet
                    p.insurance_bet = insurance_bet
                    log(f"{p.name} takes insurance ${insurance_bet:.2f}")
                else:
                    p.insurance_bet = 0.0

    def check_dealer_blackjack(self) -> bool:
        """ Dealer checks if they have blackjack """
        up = self.dealer.hands[0][0].rank
        hole = self.dealer.hands[0][1].rank
        return (up == 'A' and hole in ('10', 'J', 'Q', 'K'))

    def resolve_dealer_blackjack(self, round_start_balances):
        """ If the dealer has the blakcjack, then the round is resolved immediately"""
        # Hole card gets revealed now in real play
        self.reveal_dealer_hole_card()

        for p in self.players:
            if p.bet <= 0:
                continue

            # insurance pays 2:1 profit + stake back
            if p.insurance_bet > 0:
                p.balance += 3.0 * p.insurance_bet

            # main hand outcome vs dealer blackjack
            if p.has_blackjack(0):
                p.pushes += 1
                p.balance += p.hand_bets[0]
            else:
                p.losses += 1

        # record per-round profit correctly
        self.record_round_results(round_start_balances)

    def dealer_actions(self):
        """ Handles what the dealer does when they flip their card, based on the rules of the table """
        # dealer hole card becomes visible before dealer plays
        self.reveal_dealer_hole_card()

        while True:
            total, soft = self.dealer.hand_total(0)
            if total > 21:
                break
            if total > 17:
                break
            if total == 17:
                if DEALER_HITS_SOFT_17 and soft:
                    pass  # hit
                else:
                    break

            self.dealer.add_card_to_hand(self.deck.deal_card(seen=True))

    def player_actions(self):
        """ What all the players will do for their turn """
        for p in self.players:
            if p.bet <= 0:
                continue

            # queue of (hand_index, was_split_hand)
            queue = [(0, False)]

            while queue:
                hand_index, was_split = queue.pop(0)

                while True:
                    up = self.dealer.hands[0][0].rank
                    move = p.bestMove(up, hand_index) if p.name == "Player 1" else p.basic_best_move(up, hand_index)

                    hand = p.hands[hand_index]

                    # SPLIT
                    if move == 'SPLIT':
                        if was_split:
                            move = 'HIT'  # disallow resplitting in this model
                        if len(hand) != 2 or hand[0].rank != hand[1].rank:
                            move = 'HIT'

                    if move == 'SPLIT':
                        was_aces = (hand[0].rank == 'A')
                        c1 = self.deck.deal_card(seen=True)
                        c2 = self.deck.deal_card(seen=True)
                        ok = p.split_hand(hand_index, c1, c2)
                        if not ok:
                            move = 'HIT'
                        else:
                            new_index = len(p.hands) - 1
                            # after split, schedule both hands
                            if not was_aces:
                                queue = [(hand_index, True), (new_index, True)] + queue
                            # split aces: one card each, no further action
                            break

                    # DOUBLE
                    if move == 'DOUBLE':
                        if was_split and not ALLOW_DOUBLE_AFTER_SPLIT:
                            move = 'HIT'
                        if len(p.hands[hand_index]) != 2:
                            move = 'HIT'

                    if move == 'DOUBLE':
                        ok = p.double_down(hand_index)
                        if ok:
                            p.add_card_to_hand(self.deck.deal_card(seen=True), hand_index)
                            break
                        else:
                            move = 'HIT'

                    # HIT / STAND
                    if move == 'HIT':
                        p.add_card_to_hand(self.deck.deal_card(seen=True), hand_index)
                        if p.is_busted(hand_index):
                            break
                        continue

                    if move == 'STAND':
                        break

    def settle_round(self, round_start_balances):
        """ Handles the outcomes for everyone in the round """
        dealer_total, _ = self.dealer.hand_total(0)
        dealer_blackjack = (dealer_total == 21 and len(self.dealer.hands[0]) == 2)

        for p in self.players:
            if p.bet <= 0:
                continue

            for hi in range(len(p.hands)):
                bet = p.hand_bets[hi]
                total, _ = p.hand_total(hi)

                # blackjack payout only on original unsplit hand (your prior behavior)
                player_blackjack = (hi == 0 and len(p.hands) == 1 and p.has_blackjack(0))

                if total > 21:
                    p.losses += 1
                    continue

                if dealer_total > 21:
                    if player_blackjack:
                        p.blackjacks += 1
                        p.balance += 2.5 * bet
                    else:
                        p.wins += 1
                        p.balance += 2.0 * bet
                    continue

                if dealer_blackjack:
                    # dealer blackjack case should’ve been handled earlier, but keep safe
                    if player_blackjack:
                        p.pushes += 1
                        p.balance += bet
                    else:
                        p.losses += 1
                    continue

                if player_blackjack:
                    p.blackjacks += 1
                    p.balance += 2.5 * bet
                elif total > dealer_total:
                    p.wins += 1
                    p.balance += 2.0 * bet
                elif total < dealer_total:
                    p.losses += 1
                else:
                    p.pushes += 1
                    p.balance += bet

        self.record_round_results(round_start_balances)

    def record_round_results(self, round_start_balances):
        """ Records the outcome of the round (bankroll-wise)"""
        rr = {}
        for p in self.players:
            rr[p.name] = p.balance - round_start_balances[p.name]
        self.round_results.append(rr)

    def play(self, num_rounds=1):
        """ Plays a round of blackjack for everyone at the table """
        self.num_rounds = num_rounds

        for _ in range(num_rounds):
            if self.deck.cards_left() <= self.deck.cut_card_position:
                self.deck.initialize_deck()

            self.reset_hands()

            round_start_balances = {p.name: p.balance for p in self.players}

            self.collect_bets()

            # track "hands played" per player correctly (only if actually betting)
            for idx, p in enumerate(self.players):
                if p.bet > 0:
                    self.hands_played[idx] += 1

            self.deal_initial_cards()

            if self.check_dealer_blackjack():
                self.resolve_dealer_blackjack(round_start_balances)
                continue

            self.player_actions()
            self.dealer_actions()
            self.settle_round(round_start_balances)

            log(f"Running count: {self.deck.count}")
            log(f"True count: {self.deck.get_true_count():.2f}")
            log(f"Betting count: {self.deck.get_betting_count():.2f}")


class Simulation:
    """ Handles the main simulator for the blackjack round """
    def __init__(self, num_simulations=100, hands_per_simulation=10000):
        self.num_simulations = num_simulations
        self.hands_per_simulation = hands_per_simulation

        self.results = {}          # name -> list[final_balance]
        self.returns = {}          # name -> list[profit]
        self.hands_played = {}     # name -> list[hands played]
        self.total_stats = {}      # name -> dict cumulative

    def run_simulation(self):
        """ Runs the simulation based on the input of the game """
        for i in range(self.num_simulations):
            game = BlackjackGame(num_decks=8, num_players=3)
            game.play(num_rounds=self.hands_per_simulation)

            for idx, p in enumerate(game.players):
                self.results.setdefault(p.name, []).append(p.balance)
                self.returns.setdefault(p.name, []).append(p.balance - 500.0)
                self.hands_played.setdefault(p.name, []).append(game.hands_played[idx])

                ts = self.total_stats.setdefault(p.name, {"wins": 0, "losses": 0, "pushes": 0, "blackjacks": 0})
                ts["wins"] += p.wins
                ts["losses"] += p.losses
                ts["pushes"] += p.pushes
                ts["blackjacks"] += p.blackjacks

            if VERBOSE:
                print(f"Simulation {i+1}/{self.num_simulations} completed.")
            else:
                print(f"Sim {i+1}/{self.num_simulations} done.", end="\r")

        print()
        self.analyze_results()

    def analyze_results(self):
        """ Analyzes the results of the game and outputs the results for each player """
        print(f"\nSimulation Results after {self.num_simulations} simulations:")

        for name, balances in self.results.items():
            avg_balance = statistics.mean(balances)
            std_balance = statistics.stdev(balances) if len(balances) > 1 else 0
            med_balance = statistics.median(balances)

            rets = self.returns[name]
            avg_ret = statistics.mean(rets)
            std_ret = statistics.stdev(rets) if len(rets) > 1 else 0

            played = self.hands_played[name]
            avg_played = statistics.mean(played)
            pct_played = (avg_played / self.hands_per_simulation) * 100.0  # fixed

            ts = self.total_stats[name]
            total_hands = ts["wins"] + ts["losses"] + ts["pushes"]
            win_rate = (ts["wins"] / total_hands) * 100 if total_hands else 0
            loss_rate = (ts["losses"] / total_hands) * 100 if total_hands else 0
            push_rate = (ts["pushes"] / total_hands) * 100 if total_hands else 0
            bj_rate = (ts["blackjacks"] / total_hands) * 100 if total_hands else 0

            print(f"\n{name}:")
            print(f"  Final Balance: avg=${avg_balance:.2f}, std=${std_balance:.2f}, median=${med_balance:.2f}")
            print(f"  Return:        avg=${avg_ret:.2f}, std=${std_ret:.2f}, min=${min(rets):.2f}, max=${max(rets):.2f}")
            print(f"  Hands played:  avg={avg_played:.1f}/{self.hands_per_simulation} ({pct_played:.2f}%)")
            print(f"  Rates: win={win_rate:.2f}%, loss={loss_rate:.2f}%, push={push_rate:.2f}%, bj={bj_rate:.2f}%")

            if PLOT_RESULTS:
                plt.figure(figsize=(10, 6))
                plt.plot(rets, marker='o')
                plt.title(f"{name} Return Per Simulation")
                plt.xlabel("Simulation")
                plt.ylabel("Return ($)")
                plt.grid(axis='y')
                plt.show()


class Counter:
    @staticmethod
    def sumHand(hand_ranks):
        """ Sums the hand based on the cards involved """
        total = 0
        aces = 0
        for rank in hand_ranks:
            if rank in ('10', 'J', 'Q', 'K'):
                total += 10
            elif rank == 'A':
                total += 11
                aces += 1
            else:
                total += int(rank)

        while total > 21 and aces:
            total -= 10
            aces -= 1

        is_soft = (aces > 0)
        return total, is_soft


if __name__ == "__main__":
    # Change the settings in any ways you want
    simulation = Simulation(num_simulations=40, hands_per_simulation=1000)
    simulation.run_simulation()
