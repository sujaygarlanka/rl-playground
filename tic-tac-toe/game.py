class TicTacToe():
    def __init__(self):
        self.board = [" "] * 9
        # self.game_over = False
    
    # next_state, reward, done, win
    def step(self, action, player):
        if self.board[action] != " ":
            return None, -10, True, False
        
        self.board[action] = player
        # Win
        if self._check_win(player):
            # self.game_over = True
            return self._get_state(), 10, True, True
        # Tie
        elif " " not in self.board:
            return self._get_state(), 0, True, False
        else:
            return self._get_state(), 0, False, False
        
    def _check_win(self, player):
        win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in win_states:
            if self.board[i] == self.board[j] == self.board[k] == player:
                return True
        return False
    
    def _get_state(self):
        return "".join(self.board)
    
    def reset(self):
        self.board = [" "] * 9
        return self._get_state(), False, False
        # self.game_over = False

    def render(self):
        print("    Game Board    ")
        print(f"  {self.board[0]}  |  {self.board[1]}  |  {self.board[2]}")
        print("-----------------")
        print(f"  {self.board[3]}  |  {self.board[4]}  |  {self.board[5]}")
        print("-----------------")
        print(f"  {self.board[6]}  |  {self.board[7]}  |  {self.board[8]}")
        print("")

class SimpleTicTacToe():
    def __init__(self):
        self.board = [" "] * 3
    
    # next_state, reward, done, win
    def step(self, action, player):
        if self.board[action] != " ":
            return None, -10, True, False
        
        self.board[action] = player
        if self._check_win(player):
            return self._get_state(), 10, True, True
        elif " " not in self.board:
            return self._get_state(), 0, True, False
        else:
            return self._get_state(), 0, False, False
        
    def _check_win(self, player):
        win_states = [(0,1), (1,2)]
        for (i, j) in win_states:
            if self.board[i] == self.board[j] == player:
                return True
        return False
    
    def _get_state(self):
        return "".join(self.board)
    
    def reset(self):
        self.board = [" "] * 3
        return self._get_state(), False, False
        # self.game_over = False

    def render(self):
        print("    Game Board    ")
        print(f"  {self.board[0]}  |  {self.board[1]}  |  {self.board[2]}")
        print("")



# game = TicTacToe()
# game.render()
# game.step(1, "x")
# game.render()
# game.step(4, "x")
# game.render()
# game.step(7, "x")
# game.render()