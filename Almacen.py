#Ejercicio: https://www.youtube.com/watch?v=iKdlKYG78j4
import numpy as np
"""ENTORNO Y RECOMPENSAS"""
#Tamaño del alamacen (121 casillas 11 x 11)
environment_rows = 11
environment_cols = 11

#Matriz tridimensional, la tercera dimension la usaremos para establecer los 4 valores de cada accion (arriba, abajo, deracha, izquierda) en ese estado (localización)
q_values = np.zeros((environment_rows, environment_cols, 4))

#Codigos numericos para cada acción 0 -> arriba, 1 -> derecha, 2 -> abajo, 3 -> izquierda
actions = ['up', 'right', 'down', 'left']

#Construimos nueva matriz con la misma forma que el mapa (11 x 11) donde situaremos las recompensas por cada casilla
rewards = np.full((environment_rows, environment_cols), -100) #Iniciamos todas las casilla a -100 por que son lo q vale la mayoria simplemente
rewards[0, 5] = 100 #Meto el Area de packaging (fin)

#Para facilitar el trabajo de montar el escenario metemos las casillas por las que puede ir nuestro robot en un diccionario
aisles = {} #store locations in a dictionary
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

#Bolcamos todas las casillas que hemos metido en el diccionario en la matriz de recompensas
for row_index in range (1, 10):
    for colum_index in aisles[row_index]:
        rewards[row_index, colum_index] = -1#A las casillas "buenas" le atribuimos el valor de -1
"""ENTORNO Y RECOMPENSAS"""

"""MODELO DE ENTRENAMIENTO"""
#Esta función se encarga de determinar si el robot se encuentra en un estado termina (se ha chocado o ha llegado a la area de packaging)
def is_terminal_state(current_row_index, current_column_index):
    if(rewards[current_row_index, current_column_index] == -1):
        return False#-1 son las casillas que puede pisar, asi que seguimos
    else:
        return True#Si pisa una casilla -100 o 100 se acabo, ha llegado el final

#Esta función nos da una localizacion aleatoria valida para empezar la partida, aquí empezará el robot
def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_cols)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_cols)
    return current_row_index, current_column_index

#Funcion que nos proporciona cual serà la siguiente accion del robot
def get_next_action(current_row_index, current_column_index, epsilon):
    if(np.random.random() < epsilon):#Numero aleatorio, si mas pequeño que el epsilon (0.9)
        #Escojemos de las 4 opciones la acción que tenga mas probabilidad de salir bien de nuestra matriz de q_values (la que iniciamos al principio)
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        #De lo contrario, simplemente elejimos una acción aleatoria obligando así al robot a explorar nuevas posibilidades y que no se quede solo con una
        return np.random.randint(4)

#Funcion que nos proporciona la nueva posicion donde se moverá nuesto robot
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if(actions[action_index] == 'up' and current_row_index > 0):
        new_row_index -= 1
    elif (actions[action_index] == 'right' and current_column_index < environment_cols - 1):
        new_column_index += 1
    elif (actions[action_index] == 'down' and current_row_index < environment_rows - 1):
        new_row_index += 1
    elif (actions[action_index] == 'left' and current_column_index > 0):
        new_column_index -= 1
    return new_row_index, new_column_index
"""MODELO DE ENTRENAMIENTO"""

"""ENCONTAR EL CAMINO (Aclarar que en esta funcion no se entrena a la IA simlemente movemos nuestro robot y no plasmamos los resultados)"""
def get_shortest_path(start_row_index, start_column_index):
    #Al comenzar comprobamos que la ubicacion que nos han dado es valida, de lo contrario acaba la función
    if(is_terminal_state(start_row_index, start_column_index)):
        return []
    else:
        current_row_index, current_column_index = start_row_index ,start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        #Mientras no estemos en una posicion terminal -100 o 100 el juego continua
        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, epsilon)#Pide la nueva acción
            #Usamos la acciñon y nos movemos a la nueva casilla
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            #Añadimos a la array de resultados la nueva posición
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path
"""ENCONTAR EL CAMINO"""

"""ENTRENAR A LA IA (¡¡¡esto ya no es una función es codigo del programa!!!)"""
#Parametros del entrenamiento
epsilon = 0.95
discount_factor = 0.9#Siempre cercano al 1
learning_rate = 0.9#Siempre cercano al 1

#Entrenamos al robot por 1000 episodios
for episode in range (1000):
    #Elegimos una localizacion aleatoria, así el entreno es mas efectivo
    row_index, colum_index = get_starting_location()
    #El episodio se acaba cuando se llega a un estado terminal -100 o 100
    while not is_terminal_state(row_index, colum_index):
        #Pedimos la siguiente acción que nos la devuelve gracias a los valores de la q_table, o hay la probbilidad de que eliga el movimiento aleatorio
        action_index = get_next_action(row_index, colum_index, epsilon)
        #Guardamos las coordenadas entes de movernos para poder usarlas después para actualizar los datos de la q_table
        old_row_index, old_column_index = row_index, colum_index
        #Nos movemos segun la acción que nos haya llegado
        row_index, colum_index = get_next_location(row_index, colum_index, action_index)
        #Guardamos cual es la recompensa de la nueva coordenada
        reward = rewards[row_index, colum_index]
        #Guardamos el q_value de las coordenadas i la accion que han sido escogidas anteriormente
        old_q_value = q_values[old_row_index, old_column_index, action_index]

        """Toca ajustar los q_values que hemos usado anteriormente, para eso usamos primero:
        Formula de diferencia temporal, explicada aquí: https://youtu.be/__t2XRxXGxI?t=478
        recompensa + factor_descuento * valor_maximo_de_q_values_actual - q_value_anterior_escogido"""
        temporal_difference_equation = reward + discount_factor * np.max(q_values[row_index, colum_index]) - old_q_value
        #Una vez tenemos la diferencia temporal haremos la equacion de Bellman, LA QUAL NOS DEVOLVERÀ EL NUEVO Q_VALUE PARA NUESTRA DECISION ANTERIOR
        #q_value_anterior + (tasa_aprendizaje * diferencia_temporal)
        bellman_equation = old_q_value + (learning_rate * temporal_difference_equation)
        #Sustituimos el antiguo q_value de nuestra decision anterior por este nuevo
        q_values[old_row_index, old_column_index, action_index] = bellman_equation
print("¡Entrenamiento finalizado!")
"""ENTRENAR A LA IA"""

"""USAR IA"""
for row_index in range(11):
    for colum_index in range(11):
        if rewards[row_index, colum_index] == -100:
            print("\033[30m" + "■" + "\033[0m", end=' ')
        elif rewards[row_index, colum_index] == -1:
            print("\033[97m" + "■" + "\033[0m", end=' ')
        else:
            print("\033[94m" + "■" + "\033[0m", end=' ')
    print()

final_path = get_shortest_path(3, 9) #starting at row 3, column 9

print("\nSOLUCIÓN")
for row_index in range(11):
    for colum_index in range(11):
        if([row_index, colum_index] in  final_path):
            print("\033[92m" + "■" + "\033[0m", end=' ')
        else:
            if rewards[row_index, colum_index] == -100:
                print("\033[30m" + "■" + "\033[0m", end=' ')
            elif rewards[row_index, colum_index] == -1:
                print("\033[97m" + "■" + "\033[0m", end=' ')
            else:
                print("\033[94m" + "■" + "\033[0m", end=' ')
    print()

"""USAR IA"""