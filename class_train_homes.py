class House:
    square: float
    floor: int
    boiler: bool
    street: 'str'

    def __init__(self,square, floor, boiler, street ):
        self.square = square
        self.floor = floor
        self.boiler = boiler
        self.street = street


class Townhouse(House):
        garage: bool

        def __init__(self, square, floor, boiler, street, garage):
            super().__init__(square, floor, boiler, street)

