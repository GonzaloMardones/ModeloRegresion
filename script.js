var stopTraining

// Obtenemos la data que esta alojada en github
async function getData() {
    const datosCasasR = await fetch('https://gonzalomardones.github.io/ModeloRegresion/datos.json')
    const datosCasas = await datosCasasR.json()
    const datosLimpios = datosCasas.map(
        casa => ({
            precios: casa.Precio,
            cuartos: casa.NumeroDeCuartosPromedio
        })
    )
    .filter(casa =>(
        casa.precios != null && casa.cuartos != null ))
    
    return datosLimpios
}

// Mostramos curva de inferencia
async function verCurvaInferencia(){
    var data = await getData()
    var tensorData = await convertirDatosATensores(data)
    const {entradasMax, entradasMin, etiquetasMin, etiquetasMax} = tensorData

    const[xs, preds] = tf.tidy( ()=> {
        const xs = tf.linspace(0,1,100)
        const preds = modelo.predict(xs.reshape([100,1]))

        const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin)
        const desnormY = preds.mul(etiquetasMax.sub(etiquetasMin)).add(etiquetasMin)

        // Desnormalizamos los datos
        return [desnormX.dataSync(), desnormY.dataSync()]
    })


    const puntosPrediccion = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    })

    const puntosOriginales = data.map(d => ({
        x: d.cuartos,
        y: d.precios
        }))

    tfvis.render.scatterplot(
        {name: "Predicciones vs Originales"},
        {values: [puntosOriginales,puntosPrediccion], series: ["Originales", "Prediccion"]},
        {
            xLabel: "Cuartos",
            yLabel: "Precio",
            height: 300
        }
    )
}

// Cargamos modelo de archivos Json y Bin
async function cargarModelo(){
    const uploadJSONInput = document.getElementById('upload-json')
    const uploadWeightsInput  = document.getElementById('upload-weights')

    modelo = await tf.loadLayersModel(tf.io.browserFiles(
        [uploadJSONInput.files[0], uploadWeightsInput.files[0]]
    ))
    console.log("Modelo Cargado")
}

function visualizador(data){
    const valores = data.map( d => ({
        x: d.cuartos,
        y: d.precios
    }))
    tfvis.render.scatterplot(
        {name: "Cuartos vs Precios"},
        {values: valores},
        {
            xLabel: "Cuartos",
            yLabel: "Precio",
            height: 300
        }
    )
}

function crearModelo(){
    const modelo = tf.sequential()

     // agregar capa oculta que va a recibir 1 dato
    modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true}))

    // agregar una capa de salida que va a tener 1 sola unidad
    modelo.add(tf.layers.dense({units: 1, useBias: true}))

    return modelo 
}
// Declaramos optimizador, f(x) de perdida y KPI que evaluaremos
const optimizador = tf.train.adam()
const funcion_perdida = tf.losses.meanSquaredError
const metricas = ['mse']

async function entrenarModelo(model, inputs, labels){
    // Preparamos el entrenamiento del modelo
    model.compile({
        optimizer: optimizador,
        loss: funcion_perdida,
        metrics: metricas
    })

    const surface = { name: 'show.history live', tab: 'Training'}
    const tamanioBatch = 28
    const epochs = 50
    const history= []
    
    return await model.fit(inputs, labels,{
        tamanioBatch,
        epochs,
        shuffle: true,
        callbacks: 
        {
            onEpochEnd: (epoch, log) => {
                history.push(log)
                tfvis.show.history(surface, history, ['loss','mse'])
                if(stopTraining){
                    modelo.stopTraining=true
                }
            }
        }
    })
}

// De los archivos que tengo en el archivo necesito convertirlos a tensores ponerlos a un formato que TFJs los reconozca
function convertirDatosATensores(data){
    return tf.tidy( () => {
        tf.util.shuffle(data)
        const entradas = data.map(d => d.cuartos)
        const etiquetas = data.map(d => d.precios)
        
        const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1])
        const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1])  


    const entradasMax = tensorEntradas.max()
    const entradasMin = tensorEntradas.min()
    const etiquetasMax = tensorEtiquetas.max()
    const etiquetasMin = tensorEtiquetas.min()

    //(dato-min)/(max-min)
    const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin))
    const etiquetasNormalizadas = tensorEntradas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin))

    return{
        entradas: entradasNormalizadas,
        etiquetas: etiquetasNormalizadas,
        entradasMin,
        entradasMax,
        etiquetasMin,
        etiquetasMax
        }
    })
}

// Almacenar el modelo
async function guardarModelo(){
    const saveResult = await modelo.save('downloads://modelo-regresion')
}

var modelo
// De los archivos que tengo en el archivo necesito convertirlos a tensores ponerlos a un formato que TFJs los reconozca
async function run(){
    const data = await getData()

    visualizador(data)

    modelo = crearModelo()

    const tensorData = convertirDatosATensores(data)
    const {entradas, etiquetas} = tensorData
    await entrenarModelo(modelo, entradas, etiquetas)

}
//Ejecutamos el programa y modelo
function ejecutarModelo(){
    run()
}

