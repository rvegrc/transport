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

        # distance_target = 0
        # for coord, self_coord in zip(target_coord, self.coord):
        #     distance_target += (coord - self_coord)**2 
        # distance_target = distance_target ** 0.5

        distance_target = 0
        distance_target = [distance_target += (coord - self_coord)**2 for coord, self_coord in zip(target_coord, self.coord)]
        distance_target = distance_target ** 0.5


        if distance >= distance_target:
            self.coord = target_coord
            print (self.name, self.coord, distance_target)
            return 
        koef = distance / distance_target

        # self_coord_temp = []
        # for t_coord, self_coord in zip(target_coord, self.coord):
        #     self_coord_temp.append(self_coord + (t_coord - self_coord)*koef)
        
        # self.coord = self_coord_temp

        self.coord = [self_coord + (t_coord - self_coord)*koef for t_coord, self_coord in zip(target_coord, self.coord)]

        # list compreh  for + if  or if + for вложенность + apply lambda x: colomns... in jp 2 for and 2 if

        print (self.name, self.coord, distance)


        # дочерний класс, наследование создать и применить летающие и ползающие


class Fly(Insect):
    n_wings: int

    def __init__(self, name, weight, hight, speed, n_legs, coord, n_wings):
        super().__init__(name, weight, hight, speed, n_legs, coord)
        self.n_wings = n_wings


class Crawl(Insect):
    pass


mosquito = Fly(name='mosquito', weight=1.2, hight=1.3, speed=21.4, n_legs=6, coord=[0, 0, 0], n_wings=2)

spider = Crawl(name='spider', weight=2.2, hight=4.3, speed=1.4, n_legs=8, coord=[0, 0, 0])

# mosquito.data_print() 

# mosquito.fly([8, 1, 5], 6)


spider.data_print()

spider.distance_coords([8, 1, 0], 2)

mosquito.distance_coords([8, 1, 6], 2)




