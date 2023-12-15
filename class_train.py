class Insect:
    name: str
    weight: float
    hight: float
    speed: float
    n_legs: int
    coord: list
    
    def __init__(self, name, weight, hight, speed, n_legs, coord):
        self.name = name
        self.weight = weight
        self.hight = hight
        self.speed = speed
        self.n_legs = n_legs
        self.coord = coord
    
    def data_print(self):
        print(self.name, self.weight, self.hight, self.speed, self.n_legs, self.coord)

    def distance_coords(self, target_coord, time):
        distance = self.speed * time
        distance_target = (
            (target_coord[0] - self.coord[0])**2 
            + (target_coord[1]- self.coord[1])**2 
            + (target_coord[2]- self.coord[2])**2
            ) ** 0.5
        if distance >= distance_target:
            self.coord = target_coord
            print (self.name, self.coord, distance_target)
            return 
        koef = distance / distance_target
        self.coord = [target_coord[0]*koef, target_coord[1]*koef, target_coord[2]*koef]
        print (self.name, self.coord, distance)

        # дочерний класс, наследование создать и применить летающие и ползающие


class Fly(Insect):
    pass
    # n_wings: int

    # def __init__(self, n_wings):
    #     self.n_wings = n_wings

class Crawl(Insect):
    pass


mosquito = Fly(name='mosquito', weight=1.2, hight=1.3, speed=21.4, n_legs=6, coord=[0, 0, 0])

spider = Crawl(name='spider', weight=2.2, hight=4.3, speed=1.4, n_legs=8, coord=[0, 0, 0])

mosquito.data_print() 

# mosquito.fly([8, 1, 5], 6)

spider.data_print()

spider.distance_coords([8, 1, 0], 2)

mosquito.distance_coords([8, 1, 6], 2)


