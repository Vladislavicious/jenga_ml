# Цель проекта
Обучение алгоритма машинного обучения с подкреплением для построения **башни-дженга**.


![построенная башня](https://drive.google.com/uc?export=view&id=1GNg-RzvU0-4Mr3DkiXisQNv-1SN4OWlq)



## Задачи проекта
1. Изучить существующие алгоритмы (DQN, **PPO**, A3C и др.) и выбрать наиболее подходящий для проекта.
1. Создать базовый класс окружения с моделью отдельного блока дженга, включая физические параметры (координаты в пространстве, размеры) и действиями с блоками.
1. Разработать интерфейс взаимодействия между алгоритмом МО и средой MuJoCo для возможности получения состояния среды и отправки действий от агента. 
1. Реализовать выбранный алгоритм МО с подкреплением на Python, настроив параметры обучения (функция наград, скорость обучения, коэффициенты дисконтирования и т.д.). 
1. Обучить агента.
1. Провести тесты.  

## Инструкции по запуску
Локальный запуск на компьютере:

+ Шаг 1: Требуются установленные [Git](https://git-scm.com/install/) и [Python](https://www.python.org/downloads/).
+ Шаг 2: Клонируйте [проект](https://github.com/Vladislavicious/jenga_ml).

Откройте терминал (командную строку) и выполните:
```bash
git clone https://github.com/Vladislavicious/jenga_ml.git
cd jenga_ml
```
+ Шаг 3: Создайте виртуальное окружение.
  
*Для Windows:*
```bash
python -m venv venv
venv\Scripts\activate
```
*Для Mac/Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```

+ Шаг 4: Установите [зависимости](https://github.com/Vladislavicious/jenga_ml/blob/main/requirements.txt).
```bash
pip install -r requirements.txt
```

+ Шаг 5: Запустите Jupyter Notebook.
```bash
jupyter notebook main.ipynb
```
**Автоматически откроется браузер с блокнотом main.ipynb.**

## Реализованные функции
