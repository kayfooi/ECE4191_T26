
class World:
    def __init__(self, quadrant):
        self.balls = []
        # TODO: define world boundaries based on quadrant
        self.quadrant = quadrant



class Ball:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.collected = False
