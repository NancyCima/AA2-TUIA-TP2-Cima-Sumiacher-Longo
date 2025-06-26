# Flappy Bird
## Ingeniería de Características

### Estado crudo del entorno:

* `player_y`: posición vertical (Y) del jugador (el pájaro).
* `player_vel`: velocidad vertical actual.
* `next_pipe_dist_to_player`: distancia horizontal al próximo tubo.
* `next_pipe_top_y` / `next_pipe_bottom_y`: coordenadas Y del hueco del tubo.

Todos estos valores son continuos (en píxeles o velocidades), y para usar Q-Learning los convertimos a valores discretos simples de interpretar.


### Características del estado procesadas y discretizadas:

El estado continuo se transforma en una tupla de 3 números enteros:

1. **¿Qué tan lejos está el próximo tubo horizontalmente?**

   **Idea**: Si el tubo está muy cerca, tengo que reaccionar rápido. Si está lejos, tengo más margen.

   **Simplificación**: Dividimos la distancia en 10 zonas. Las zonas cercanas al tubo tienen más detalle para aprender con precisión cuándo y cómo saltar.

   **Importancia**: Indica cuán urgente es tomar una acción. Si el tubo está cerca, conviene actuar con precisión.

---

2. **¿Estoy bien alineado con el hueco del tubo?**

   **Idea**: ¿Estoy por encima, centrado o por debajo del centro del hueco?

   **Simplificación**: Calculamos la diferencia entre mi posición (`player_y`) y el centro del hueco. Esa diferencia se divide en 12 zonas: muchas más zonas cuando estoy cerca del centro del hueco (más detalle), y menos cuando estoy lejos (menos crítico).

   **Importancia**: Esta variable ayuda a saber si hay que subir o mantenerse estable para entrar al hueco.

---

3. **¿Cuál es mi velocidad vertical?**

   **Idea**: ¿Estoy cayendo rápido, flotando, o subiendo?

   **Simplificación**: Tomamos la velocidad vertical (que puede ir de -12 a 12) y la dividimos en 6 rangos. Esto captura si el pájaro está acelerando, frenando, cayendo libremente, etc.

   **Importancia**: Saber cómo me estoy moviendo ahora permite elegir si conviene hacer un salto o esperar.

---

### Ejemplo de estado discreto:

* "El tubo está a distancia media" (Componente 1)
* "Estoy un poco por debajo del centro del hueco" (Componente 2)
* "Estoy cayendo lentamente" (Componente 3)

## Conclusiones y Análisis

### Agente Q-Learning
El agente Q-Learning fue capaz de aprender a jugar razonablemente bien gracias a:

* Una discretización inteligente que da más detalle en las zonas críticas (como cuando el tubo está cerca o el pájaro está cerca del centro del hueco).
* Un sistema de exploración adaptativo que prueba más acciones en estados poco visitados.
* Bonificaciones en la recompensa por mantenerse cerca del centro del hueco, incentivando un vuelo más seguro.

#### Ventajas:

* Rápido de entrenar y estable.
* Buen rendimiento sin necesidad de red neuronal.
* Capacidad de interpretar el estado de forma lógica y reducir errores tempranos.

#### Limitaciones:

* Al ser discreto, no puede generalizar bien entre estados similares pero distintos.
* Puede volverse ineficiente con más variables o más detalle.
* Necesita muchos episodios para visitar suficientes combinaciones de estado y acción.

### Agente DQ-Learning  
El agente DQ-Learning usa una red neuronal para aprender a partir de los valores Q directamente, en lugar de almacenarlos en una tabla. Sin embargo, en este caso, su desempeño fue inferior al del agente tabular, probablemente debido a la limitada cantidad de datos de entrenamiento (814 estados).

#### Ventajas:

* Generaliza: puede comportarse en estados no vistos, aunque con limitaciones dadas las pocas muestras.  
* Ahorra memoria: no necesita guardar una tabla gigante.  
* Fácil de usar: se guarda como un único archivo entrenado.

#### Limitaciones:

* Rendimiento inferior al agente Q-Learning tradicional en este entorno.  
* Sensible a la cantidad y calidad de datos de entrenamiento.  
* Menor fluidez en ejecución, con caída notable de FPS.  
* No logra superar al agente tabular con la configuración y datos actuales.

#### Observaciones:

* Se entrenó con un MSE cercano a 0.75, lo que indica que la red aproximó la Q-table, pero con margen importante de error.  
* El agente tiende a comportarse peor y de forma menos estable que el Q-Agent. La máxima recompensa vista obtenida fue de 30 tuberias.  
* Más datos y un estado con mayor detalle podrían mejorar el rendimiento de la red.

## Conclusión Final  
Tanto el agente Q-Learning como el DQ-Learning lograron aprender a jugar, pero el Q-Agent tradicional mostró un rendimiento superior y mayor fluidez en tiempo real. La red neuronal, aunque prometedora por su capacidad de generalización, quedó limitada por la escasez de datos de entrenamiento y no logró mejorar la experiencia general.

Una mejora clave para ambos enfoques sería utilizar un estado más detallado — más bins o más variables del entorno (por ejemplo, información del segundo tubo). Esto permitiría una mejor representación del entorno, mejor aprendizaje y decisiones más acertadas. Además, se obtendrían más datos para entrenar, lo que ayudaría especialmente a la red neuronal a obtener una generalización mayor y mejorar su rendimiento.
