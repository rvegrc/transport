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


        # дочерний класс, наследование создать и применить летающие и ползающие


class Fly(Insect):

    def fly(self, target_coord, time):
        distance = self.speed * time
        distance_target = (
            (target_coord[0] - self.coord[0])**2 
            + (target_coord[1]- self.coord[1])**2 
            + (target_coord[2]- self.coord[2])**2
            ) ** 0.5
        if distance >= distance_target:
            self.coord = target_coord
            print (self.coord, distance)
            return 
        koef = distance / distance_target
        # self.coord = [target_coord[0]*koef, target_coord[1]*koef, target_coord[2]*koef]


class Crawl(Insect):
    def crawl(self, target_coord, time):
        distance = self.speed * time
        distance_to_target = ( 
            (target_coord[0]**2 + target_coord[1]**2)**0.5
            -(self.coord[0]**2 + self.coord[1]**2)**0.5
        )
        if distance >= distance_to_target:
            self.cood = target_coord
            return





mosquito = Insect(name='mosquito', weight=1.2, hight=1.3, speed=21.4, n_legs=6, coord=[0, 0, 0])

spider = Insect(name='spider', weight=2.2, hight=4.3, speed=1.4, n_legs=8, coord=[0, 0, 0])

# mosquito.data_print() 

mosquito.fly([8, 1, 5], 6)



