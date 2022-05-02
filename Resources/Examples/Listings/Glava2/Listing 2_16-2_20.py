# Класс кошки
class Cat:
    Name_Class = "Кошки"

    # Действия, которые надо выполнять при создании объекта "Кошка"
    def __init__(self, wool_color, eyes_color, name):
        self.wool_color = wool_color
        self.eyes_color = eyes_color
        self.name = name

    # Мурлыкать
    def purr(self):
        print("Муррр!")

    # Шипеть
    def hiss(self):
        print("Шшшш!")

    # Царапаться
    def scrabble(self):
        print("Цап-царап!")


my_cat = Cat('Рыжий', 'Голубые', 'Мурка')
my_cat.name = "Васька"
my_cat.wool_color = "Черный"
print("Наименование класса - ", my_cat.Name_Class)
print("Вот наша кошка:")
print("Цвет шерсти- ", my_cat.wool_color)
print("Цвет глаз- ", my_cat.eyes_color)
print("Кличка- ", my_cat.name)
my_cat.purr()

my_cat = Cat('Белая', 'Зеленые', 'Мурка')
my_cat.name = "Васька"
my_cat.wool_color = "Черный"
print("Наименование класса - ", my_cat.Name_Class)
print("Вот наша кошка:")
print("Цвет шерсти- ", my_cat.wool_color)
print("Цвет глаз- ", my_cat.eyes_color)
print("Кличка- ", my_cat.name)
