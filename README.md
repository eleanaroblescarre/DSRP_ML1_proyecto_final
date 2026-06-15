# Proyecto Final - Customer Segmentation

## a. Problema de ML
El reto consiste en segmentar clientes usando sus datos de facturación. Queremos agruparlos en categorías que reflejen su comportamiento de compra, para luego poder diseñar estrategias de marketing más efectivas.

## b. Diagrama de Flujo
![Diagrama RFM KMeans](/Diagrama_Flujo.png)

## c. Dataset
Fuente: Kaggle - Customer Segmentation Dataset  
Registros: 540k  
Variables: datos demográficos, historial de compras, categorías de productos.  
Formato: CSV

## d. Model Card
Detalles del modelo
- Tipo: K-Means clustering
- Librería: scikit-learn
- Variables usadas: CustomerInvoiceTotal, CustomerInvoiceCount, RecencyDays
- Fecha: Junio 2026

Uso previsto
- Objetivo: dividir a los clientes en grupos según su comportamiento de compra.
- Contexto: análisis de marketing y fidelización.
- Quién lo usaría: equipos de estrategias, marketing o intelignecia commercial.

Factores considerados
- Recency: cuántos días pasaron desde la última compra.
- Frequency: cuántas facturas tiene cada cliente.
- Monetary: cuánto gastó en total.
- Se filtraron facturas canceladas para no distorsionar los resultados.

Métricas
- Silhouette Score: 0.78 → bastante alto, lo que indica que los clusters están bien separados.
- Número de clusters elegido: 2.
- Interpretación:
  - Cluster 0 → clientes de bajo valor (menos gasto, menos facturas, más tiempo desde la última compra).
  - Cluster 1 → clientes de alto valor (más gasto, más facturas, compras recientes).

Datos de evaluación
- Dataset: facturas de clients (invoices).
- Preprocesamiento: 
  - conversión de intervalos de tiempo a días,
  - normalización de variables numéricas,
  - exclusión de facturas canceladas.

Consideraciones éticas
- El modelo solo usa comportamiento de compra, no datos sensibles como edad o género.
- No debe usarse para discriminar clientes, sino para entender patrones de consumo.
- Los clusters son una guía. Se pueden complementar con reglas de negocio y conocimiento de expertos.

Limitaciones y recomendaciones
- El modelo encontró 2 grupos principales; si se require más detalle se debe evaluar otras variables fuera de un análisis RFM.
- Los resultados dependen de la calidad del dataset, por lo que conviene actualizarlo periódicamente.
- Ideal complementar con visualizaciones (ej. boxplots) para explicar mejor las diferencias entre clusters.


## e. Resultados
1. Se estimó que 2 clusters era lo mas optimo
2. El modelo K-Means alcanzó un Silhouette Score de 0.78, lo que indica una buena separación entre clusters.


## f. Conclusiones

Cluster 0 (clientes de bajo valor):
 - Menor gasto total en facturas.
 - Menor número de facturas emitidas.
 - Intervalos de tiempo más largos desde la última compra (clientes menos recientes).
 - Representan clientes con baja frecuencia y menor contribución al ingreso, candidatos para estrategias de reactivación o campañas de retención.
 - Este Cluster puede ser gestionado con campañas de reactivación, descuentos personalizados, recordatorios de compra.

Cluster 1 (clientes de alto valor):
 - Mayor gasto total en facturas, con mayor dispersión y presencia de outliers.
 - Mayor número de facturas emitidas, reflejando alta frecuencia de compra.
 - Intervalos de tiempo más cortos desde la última compra (clientes más recientes).
 - Constituyen el segmento más valioso, ideal para programas de fidelización y estrategias de mantenimiento de relación.
 - Los clients de este cluster pueden ser gestionados con programas de lealtad, beneficios exclusivos, estrategias de upselling y cross-selling.

