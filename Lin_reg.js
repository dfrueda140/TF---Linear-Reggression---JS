//Initialize the vectors to put the data of the dots
let x_vals = [];
let y_vals = [];

//Initialize the values for slope and y interception
let m,b; 

//optimizador de las variables. 
const learningRate = 0.4;
const optimizer = tf.train.sgd(learningRate);


function setup(){
    createCanvas(800,400);
    //inivitalize the tensors for slope and y interception. 
    m =  tf.variable(tf.scalar(random(1)));
    b =  tf.variable(tf.scalar(random(1)));
}

function predict(x){
    const xs = tf.tensor1d(x);
    // Y= mx + b
    const ys = xs.mul(m).add(b);
    return ys;
}

function mousePressed(){
    // map the values in weight and height to a value between 0 and 1.
    let x = map(mouseX, 0 , width , 0 , 1);
    let y = map(mouseY, 0 , height, 1 , 0);
    // store value in the vectors.
    x_vals.push(x);
    y_vals.push(y);
}

function loss(pred,labels){
    return pred.sub(labels).square().mean();
}

function draw(){
    //condition the beguining to 
    tf.tidy(()=> {
        if(x_vals.length>0){
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(()=>loss(predict(x_vals),ys));
        }
    });

    background(0);
    stroke(255);
    strokeWeight(4);
    //Create the dots of the values store in the Xs and Ys
    for(let i = 0; i < x_vals.length; i++){
        let px =  map(x_vals[i], 0,1,0      ,width);
        let py =  map(y_vals[i], 0,1,height ,0);
        point(px,py);
    }

    // draw the line using the predict function to calculate Ys
    tf.tidy(()=>{
    const xs = [0,1];
    const ys = predict(xs);
    let lineY = ys.dataSync();
    //view the results of Y changing
    //ys.print()

    let x1 = map(xs[0], 0, 1, 0, width);
    let x2 = map(xs[1], 0, 1, 0, width);
    
    
    let y1 = map(lineY[0], 0, 1, height, 0);
    let y2 = map(lineY[1], 0, 1, height, 0);

    line(x1,y1,x2,y2);

    });
    console.log(tf.memory().numTensors);
}




