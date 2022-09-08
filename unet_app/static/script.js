var imageCanvas = document.getElementById("imageCanvas");
var imageCtx = imageCanvas.getContext("2d");
imageCtx.imageSmoothingEnabled = false;

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

var predCanvas = document.getElementById("predCanvas");
var predCtx = predCanvas.getContext("2d");
predCtx.imageSmoothingEnabled = false;

var cursorCanvas = document.getElementById("cursorCanvas");
var cursorCtx = cursorCanvas.getContext("2d");
cursorCtx.imageSmoothingEnabled = false;

var stroke = [];
var annotationData = [];

var ix = "";
var iy = "";

var brushSize = 40;

var cursorX = "";
var cursorY = "";

var leftDown = false;
var rightDown = false;
var shiftDown = false;
var ctrlDown = false;

var numClasses = 2;
var labelClass = 1;

var centerX = 350;
var centerY = 350;

var sliceLock = false;

var predDisplayed = false;

var sliceImage = new Image();
var predImage = new Image();

var colors = ['rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)', 'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)', 'rgba(245, 130, 48, 1)',
              'rgba(145, 30, 180, 1)', 'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)', 'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)'];

imageCtx.translate(centerX, centerY);
ctx.translate(centerX, centerY);
predCtx.translate(centerX, centerY);
cursorCtx.translate(centerX, centerY);

// [a, b, c, d, e, f] = [x_scale, x_rot, y_rot, y_scale, x_shift, y_shift]
var t = ctx.getTransform();

updateColor(colors[labelClass]);
ctx.lineWidth = brushSize;
ctx.lineCap = "round";

// Sampler settings --------------------------------------------------------------------------------------
document.getElementById("inputSizeSelector").addEventListener("change", function() {

    resetTransform();
    stroke = [];
    annotationData = [];

    var inputSize = document.getElementById("inputSizeSelector").value;

    $.ajax({
        type: "POST",
        url: "/update_input_size",            
        data: JSON.stringify({'inputSize': inputSize}),
        dataType: "json",
        success: function (result) {
            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
            redraw();
        }
    });
});


document.getElementById("samplingModeSelector").addEventListener("change", function() {

    resetTransform();
    stroke = [];
    annotationData = [];

    var samplingMode = document.getElementById("samplingModeSelector").value;

    $.ajax({
        type: "POST",
        url: "/update_sampling_mode",            
        data: JSON.stringify({'samplingMode': samplingMode}),
        dataType: "json",
        success: function (result) {
            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
            redraw();
        }
    });
});

document.getElementById("clearDataButton").addEventListener("click", function() {
    $.ajax({
        type: "POST",
        url: "/clear_data",
        dataType: "json",
        success: function (result) {

            // Update sample and dataset info displays
            document.getElementById("sampleInfoLabel").innerHTML = "Current: " + result['currentSample'];
            info = "Training: " + result['numTrainSamples'] + ", Validation: " + result['numValSamples'];
            document.getElementById("datasetInfoLabel").innerHTML = info;

            // Enable elements that are no longer permanent
            document.getElementById("inputSizeSelector").disabled = false;
            document.getElementById("numClassesSelector").disabled = false;
            redraw();
            updatePlot();
        }
    });
});

document.getElementById("predictVolumesButton").addEventListener("click", function() {
    $.ajax({
        type: "POST",
        url: "/predict_volumes",            
        dataType: "json",
        success: function (result) {
            document.getElementById("trainingStatusDiv").innerHTML = "Status: Predicting volumes...";
            document.getElementById("beginTrainingButton").disabled = true;
            document.getElementById("continueTrainingButton").disabled = true;
            document.getElementById("predictVolumesButton").disabled = true;
            getPredictionProgress();
        }
    });
});

function getPredictionProgress() {
    $.ajax({
        type: "POST",
        url: "/check_prediction_status",
        dataType: "json",
        success: function (result) {
            if (result['predicting'] == 'false') {
                document.getElementById("trainingStatusDiv").innerHTML = "Status: Waiting...";
                document.getElementById("beginTrainingButton").disabled = false;
                document.getElementById("continueTrainingButton").disabled = false;
                document.getElementById("predictVolumesButton").disabled = false;
                updatePlot();
            } else {
                document.getElementById("trainingStatusDiv").innerHTML = "Status: Predicting volumes...";
                document.getElementById("beginTrainingButton").disabled = true;
                document.getElementById("continueTrainingButton").disabled = true;
                document.getElementById("predictVolumesButton").disabled = true;
                // rerun in 3 seconds
                setTimeout(function() {
                    getPredictionProgress();
                }, 3000);
            }
        }
    });
}

// Annotator settings ------------------------------------------------------------------------------------
document.getElementById("randomizeButton").addEventListener("click", function() {

    resetTransform();
    stroke = [];
    annotationData = [];

    $.ajax({
        type: "POST",
        url: "/randomize",            
        dataType: "json",
        success: function (result) {
            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];

            predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
            predDisplayed = false;
            redraw();
        }
    });
});

// document.getElementById("saveButton").addEventListener("click", function() {

//     resetTransform();
//     stroke = [];
//     annotationData = [];

//     // Disable elements that are now permanent
//     document.getElementById("inputSizeSelector").disabled = true;
//     document.getElementById("numClassesSelector").disabled = true;
//     var imageDataURL = canvas.toDataURL();
    
//     $.ajax({
//         type: "POST",
//         url: "/save_sample",            
//         data: {imageData: imageDataURL},
//         dataType: "json",
//         success: function (result) {
            
//             // Update sample and dataset info displays
//             document.getElementById("sampleInfoLabel").innerHTML = "Current: " + result['currentSample'];
//             info = "Training: " + result['numTrainSamples'] + ", Validation: " + result['numValSamples'];
//             document.getElementById("datasetInfoLabel").innerHTML = info;

//             sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
        
//             predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
//             predDisplayed = false;
//             redraw();
//         }
//     });
// });

document.getElementById("saveTrainingButton").addEventListener("click", function() {

    resetTransform();
    stroke = [];
    annotationData = [];

    // Disable elements that are now permanent
    document.getElementById("inputSizeSelector").disabled = true;
    document.getElementById("numClassesSelector").disabled = true;
    var imageDataURL = canvas.toDataURL();
    
    $.ajax({
        type: "POST",
        url: "/save_training_sample",            
        data: {imageData: imageDataURL},
        dataType: "json",
        success: function (result) {
            
            // Update sample and dataset info displays
            document.getElementById("sampleInfoLabel").innerHTML = "Current: " + result['currentSample'];
            info = "Training: " + result['numTrainSamples'] + ", Validation: " + result['numValSamples'];
            document.getElementById("datasetInfoLabel").innerHTML = info;

            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
        
            predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
            predDisplayed = false;
            redraw();
        }
    });
});

document.getElementById("saveValidationButton").addEventListener("click", function() {

    resetTransform();
    stroke = [];
    annotationData = [];

    // Disable elements that are now permanent
    document.getElementById("inputSizeSelector").disabled = true;
    document.getElementById("numClassesSelector").disabled = true;
    var imageDataURL = canvas.toDataURL();
    
    $.ajax({
        type: "POST",
        url: "/save_validation_sample",            
        data: {imageData: imageDataURL},
        dataType: "json",
        success: function (result) {
            
            // Update sample and dataset info displays
            document.getElementById("sampleInfoLabel").innerHTML = "Current: " + result['currentSample'];
            info = "Training: " + result['numTrainSamples'] + ", Validation: " + result['numValSamples'];
            document.getElementById("datasetInfoLabel").innerHTML = info;

            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
        
            predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
            predDisplayed = false;
            redraw();
        }
    });
});

document.getElementById("predictButton").addEventListener("click", function() {
    $.ajax({
        type: "POST",
        url: "/predict_sample",               
        dataType: "json",
        success: function (result) {
            predImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
            predDisplayed = true;
            redraw();
        }
    });
});

document.getElementById("togglePredictionButton").addEventListener("click", function() {
    if (predDisplayed == true) {
        var frameX = -t.e / t.a;
        var frameY = -t.f / t.d;
        var frameWidth = 2 * canvas.width / t.a;
        var frameHeight = 2 * canvas.height / t.d;
        predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
        predDisplayed = false;
    } else {
        predDisplayed = true;
    }
    redraw();
});

document.getElementById("annotationOpacitySlider").addEventListener("change", function() {
    var opacity = document.getElementById("annotationOpacitySlider").value / 100;
    document.getElementById("canvas").style = "opacity: "+opacity+";";
});

document.getElementById("predictionOpacitySlider").addEventListener("change", function() {
    var opacity = document.getElementById("predictionOpacitySlider").value / 100;
    document.getElementById("predCanvas").style = "opacity: "+opacity+";";
});





// Training settings -------------------------------------------------------------------------------------
document.getElementById("numClassesSelector").addEventListener("change", function() {
    
    resetTransform();
    stroke = [];
    annotationData = [];

    numClasses = document.getElementById("numClassesSelector").value;

    $.ajax({
        type: "POST",
        url: "/update_num_classes",            
        data: JSON.stringify({'numClasses': numClasses}),
        dataType: "json",
        success: function (result) {
            redraw();
        }
    });
});

document.getElementById("learningRateSelector").addEventListener("change", function() {
    learningRate = document.getElementById("learningRateSelector").value;
    $.ajax({
        type: "POST",
        url: "/update_learning_rate",            
        data: JSON.stringify({'learningRate': learningRate}),
        dataType: "json",
        success: function (result) {}
    });
});

document.getElementById("batchSizeSelector").addEventListener("change", function() {
    batchSize = document.getElementById("batchSizeSelector").value;
    $.ajax({
        type: "POST",
        url: "/update_batch_size",            
        data: JSON.stringify({'batchSize': batchSize}),
        dataType: "json",
        success: function (result) {}
    });
});

document.getElementById("numEpochsSelector").addEventListener("change", function() {
    numEpochs = document.getElementById("numEpochsSelector").value;
    $.ajax({
        type: "POST",
        url: "/update_num_epochs",            
        data: JSON.stringify({'numEpochs': numEpochs}),
        dataType: "json",
        success: function (result) {}
    });
});

document.getElementById("beginTrainingButton").addEventListener("click", function() {
    $.ajax({
        type: "POST",
        url: "/train_model",            
        data: JSON.stringify({'continue': 'false'}),
        dataType: "json",
        success: function (result) {
            document.getElementById("trainingStatusDiv").innerHTML = "Status: Training...";
            document.getElementById("beginTrainingButton").disabled = true;
            document.getElementById("continueTrainingButton").disabled = true;
            getTrainingProgress();
        }
    });
});

document.getElementById("continueTrainingButton").addEventListener("click", function() {
    $.ajax({
        type: "POST",
        url: "/train_model",            
        data: JSON.stringify({'continue': 'true'}),
        dataType: "json",
        success: function (result) {
            document.getElementById("trainingStatusDiv").innerHTML = "Status: Training...";
            document.getElementById("beginTrainingButton").disabled = true;
            document.getElementById("continueTrainingButton").disabled = true;
            getTrainingProgress();
        }
    });
});

function getTrainingProgress() {
    $.ajax({
        type: "POST",
        url: "/check_train_status",
        dataType: "json",
        success: function (result) {
            if (result['training'] == 'false') {
                document.getElementById("trainingStatusDiv").innerHTML = "Status: Waiting...";
                document.getElementById("beginTrainingButton").disabled = false;
                document.getElementById("continueTrainingButton").disabled = false;
                document.getElementById("predictVolumesButton").disabled = false;
                updatePlot();
            } else {
                document.getElementById("trainingStatusDiv").innerHTML = "Status: Training...";
                document.getElementById("beginTrainingButton").disabled = true;
                document.getElementById("continueTrainingButton").disabled = true;
                document.getElementById("predictVolumesButton").disabled = true;
                // rerun in 3 seconds
                setTimeout(function() {
                    getTrainingProgress();
                }, 3000);
            }
        }
    });
}

// function updatePlot() {
//     var metric = document.getElementById("metricSelector").value;
//     var includeTrainingCheckbox = document.getElementById("includeTrainingCheckbox");
//     var checked = 'true';
//     if (includeTrainingCheckbox.checked != true) {
//         checked = "false";
//     }  
//     $.ajax({
//         type: "POST",
//         url: "/update_plot",            
//         data: JSON.stringify({'metric': metric,
//                               'includeTraining': checked}),
//         dataType: "json",
//         success: function (result) {
//             trainingPlotImage = document.getElementById("trainingPlotImage");
//             trainingPlotImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
//         }
//     });
// }

// document.getElementById("metricSelector").addEventListener("change", function() {
//     updatePlot();
// });

// document.getElementById("includeTrainingCheckbox").addEventListener("change", function() {
//     updatePlot();
// });


function resetTransform() {
    imageCtx.scale(1/t.a, 1/t.d);
    ctx.scale(1/t.a, 1/t.d);
    predCtx.scale(1/t.a, 1/t.d);
    cursorCtx.scale(1/t.a, 1/t.d);

    t = ctx.getTransform();

    imageCtx.translate(-t.e + centerX, -t.f + centerY);
    ctx.translate(-t.e + centerX, -t.f + centerY);
    predCtx.translate(-t.e + centerX, -t.f + centerY);
    cursorCtx.translate(-t.e + centerX, -t.f + centerY);
    
    t = ctx.getTransform();
    redraw();
}

function updateColor(color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    cursorCtx.strokeStyle = color;
    cursorCtx.fillStyle = color;
};

function redrawImage() {
    //Draw slice image
    var ratio = sliceImage.width / sliceImage.height
    imageCtx.drawImage(sliceImage, 0, 0, sliceImage.width, sliceImage.height, -centerX, -centerY, canvas.width, canvas.width / ratio);
    if (predDisplayed == true) {
        predCtx.drawImage(predImage, 0, 0, sliceImage.width, sliceImage.height, -centerX, -centerY, canvas.width, canvas.width / ratio);
    }
};

function clearCanvas() {
    // Clear canvas
    var frameX = -t.e / t.a;
    var frameY = -t.f / t.d;
    var frameWidth = 2 * canvas.width / t.a;
    var frameHeight = 2 * canvas.height / t.d;
    imageCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
    predCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
    ctx.clearRect(frameX, frameY, frameWidth, frameHeight);
};

function redrawCursor() {
    // Draw cursor
    var frameX = -t.e / t.a;
    var frameY = -t.f / t.d;
    var frameWidth = 2 * canvas.width / t.a;
    var frameHeight = 2 * canvas.height / t.d;
    cursorCtx.clearRect(frameX, frameY, frameWidth, frameHeight);
    cursorCtx.lineWidth = 1.5;
    cursorCtx.beginPath();
    cursorCtx.arc(cursorX, cursorY, brushSize/2, 0, 2 * Math.PI);
    cursorCtx.fill();
};

function setColorFromLabel(label) {
    updateColor(colors[label]);
};

function redrawAnnotations() {
    // Draw annotations
    var i;
    for (i = 0; i < annotationData.length; i++) {
        var j;
        for (j = 0; j < annotationData[i].length; j++) {

            setColorFromLabel(annotationData[i][j]['label_class']);
            
            ctx.lineWidth = annotationData[i][j]['brush_size'];
            ctx.beginPath();
            ctx.moveTo(annotationData[i][j]['ix'], annotationData[i][j]['iy']);
            ctx.lineTo(annotationData[i][j]['x'], annotationData[i][j]['y']);
            ctx.stroke();
        }
    }
    setColorFromLabel(labelClass);
};

function redraw() {
    clearCanvas();
    ctx.save();
    redrawImage();
    redrawAnnotations();
    ctx.restore();
};

function addPath(labelClass, startX, startY, endX, endY) {
    annotationData[annotationData.length - 1].push({'label_class' : labelClass,
                                'ix' : startX / t.a - t.e / t.a,
                                'iy' : startY / t.d - t.f / t.d,
                                'x' : endX / t.a - t.e / t.a,
                                'y' : endY / t.d - t.f / t.d,
                                'brush_size' : ctx.lineWidth});
};

canvas.addEventListener("mousedown", function(event) {
    event.preventDefault()

    ix = event.offsetX;
    iy = event.offsetY;

    switch (event.which) {
        case 1:
            // alert('Left Mouse button pressed.');
            if (!shiftDown) {
                stroke = [];
                annotationData.push(stroke)
                addPath(labelClass, ix, iy, ix, iy);
            }
            leftDown = true;
            rightDown = false;
            break;
        case 2:
            // alert('Middle Mouse button pressed.');
            break;
        case 3:
            // alert('Right Mouse button pressed.');
            if (!shiftDown) {
                stroke = [];
                annotationData.push(stroke)
                addPath(0, ix, iy, ix, iy);
            }
            rightDown = true;
            leftDown = false;
            break;
        default:
            // alert('Default');
    }
    redraw();
});

canvas.addEventListener("mouseup", function(event) {
    event.preventDefault()
    switch (event.which) {
        case 1:
            // alert('Left Mouse button released.');
            leftDown = false;
            break;
        case 2:
            // alert('Middle Mouse button released.');
            break;
        case 3:
            // alert('Right Mouse button released.');
            rightDown = false;
            break;
        default:
            // alert('You have a strange Mouse!');
    }
});

canvas.addEventListener("mousemove", function(event) {
    event.preventDefault()
    
    var x = event.offsetX;
    var y = event.offsetY;

    cursorX = x / t.a - t.e / t.a;
    cursorY = y / t.d - t.f / t.d;

    if (leftDown && !shiftDown) {
        addPath(labelClass, ix, iy, x, y);
    }

    if (rightDown && !shiftDown) {
        addPath(0, ix, iy, x, y);
    }

    if (leftDown && shiftDown) {
        translateX = (x-ix) / t.a;
        translateY = (y-iy) / t.d;
        imageCtx.translate(translateX, translateY);
        ctx.translate(translateX, translateY);
        predCtx.translate(translateX, translateY);
        cursorCtx.translate(translateX, translateY);
        t = ctx.getTransform();
    }

    redrawCursor();
    redraw();

    ix = x;
    iy = y;
});

canvas.addEventListener("wheel", function(event) {
    event.preventDefault();

    var wheelDelta = ((event.deltaY || -event.wheelDelta || event.detail) >> 10) || 1;

    if (wheelDelta < 0) {
        wheelDelta = 1;
    } else if (wheelDelta > 0) {
        wheelDelta = -1;
    }

    if (shiftDown) {
        scale = 1 + 0.1 * wheelDelta;

        brushSize *= 1/scale;
        brushSize = 2 * Math.round(brushSize / 2);
        ctx.lineWidth = brushSize;

        var realX = ix / t.a - t.e / t.a;
        var realY = iy / t.d - t.f / t.d;

        if (scale > 1) {
            translateX = -0.1 * realX;
            translateY = -0.1 * realY;
            imageCtx.translate(translateX, translateY);
            ctx.translate(translateX, translateY);
            predCtx.translate(translateX, translateY);
            cursorCtx.translate(translateX, translateY);
        }
        if (scale < 1) {
            translateX = 0.1 * realX;
            translateY = 0.1 * realY;
            imageCtx.translate(translateX, translateY);
            ctx.translate(translateX, translateY);
            predCtx.translate(translateX, translateY);
            cursorCtx.translate(translateX, translateY);
        }

        imageCtx.scale(scale, scale);
        ctx.scale(scale, scale);  
        predCtx.scale(scale, scale);  
        cursorCtx.scale(scale, scale);  

        t = ctx.getTransform();
    } else if (ctrlDown) {
        if (wheelDelta > 0) {
            labelClass = labelClass + 1;
            if (labelClass > (numClasses - 1)) {
                labelClass = 1;
            }
        } else if (wheelDelta < 0) {
            labelClass = labelClass - 1;
            if (labelClass < 1) {
                labelClass = (numClasses - 1);
            }
        }
        setColorFromLabel(labelClass);
    } else {
        brushSize *= 1 + 0.1 * wheelDelta;
        //brushSize = 2 * Math.round(brushSize / 2);
        brushSize = 2 * brushSize / 2;
        ctx.lineWidth = brushSize;
    }
    redraw();
    redrawCursor();
});

$(window).keydown(function (event) {

    if (event.keyCode == 16) {
        event.preventDefault();
        shiftDown = true;
    }

    if (event.keyCode == 17) {
        event.preventDefault();
        ctrlDown = true;
    }

    if (event.ctrlKey && event.key == 'z') {
        event.preventDefault();
        annotationData.pop();
        redraw();
    }

    if (event.keyCode == 81) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: -1,
                                      y: 0,
                                      z: 0}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });
            sliceLock = false;
        }
    }
    if (event.keyCode == 69) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: 1,
                                      y: 0,
                                      z: 0}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });
            sliceLock = false;
        }
    }
    if (event.keyCode == 65) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: 0,
                                      y: -1,
                                      z: 0}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });
            sliceLock = false;
        }
    }
    if (event.keyCode == 68) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: 0,
                                      y: 1,
                                      z: 0}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });
            sliceLock = false;
        }
    }
    if (event.keyCode == 83) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: 0,
                                      y: 0,
                                      z: -1}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });
            sliceLock = false;
        }
    }
    if (event.keyCode == 87) {
        if (!sliceLock) {
            sliceLock = true;
            $.ajax({
                type: "POST",
                url: "/shift_origin",
                data: JSON.stringify({x: 0,
                                      y: 0,
                                      z: 1}),
                dataType: "json",
                success: function (result) {             
                    sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
                    redraw();
                }
            });

            sliceLock = false;
        }
    }
});

$(window).keyup(function (event) {
    if (event.keyCode == 16) {
        event.preventDefault();
        shiftDown = false;
    }
    if (event.keyCode == 17) {
        event.preventDefault();
        ctrlDown = false;
    }
});

// Strange workaround to get canvas to refresh. Will replace this when I find a better way
//------------------------------------------------------------------------------------
window.addEventListener("mousemove", function(event) {
    redraw();
    redrawCursor();
});

window.addEventListener("mouseup", function(event) {
    redraw(); 
    redrawCursor();
});

window.addEventListener("mousedown", function(event) {
    redraw();
    redrawCursor();
});
//------------------------------------------------------------------------------------

window.onload = function() {

    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        url: "/check_parameters",
        dataType: "json",
        success: function (result) {

            // Update sample and dataset info displays
            document.getElementById("sampleInfoLabel").innerHTML = "Current: " + result['currentSample'];
            info = "Training: " + result['numTrainSamples'] + ", Validation: " + result['numValSamples'];
            document.getElementById("datasetInfoLabel").innerHTML = info;
            

            document.getElementById("inputSizeSelector").value = result['inputSize'];
            document.getElementById("numClassesSelector").value = result['numClasses'];
            numClasses = result['numClasses'];

            if (result['lockParameters'] == 'true') {
                // Disable elements that are now permanent
                document.getElementById("inputSizeSelector").disabled = true;
                document.getElementById("numClassesSelector").disabled = true;
            }
        }
    });

    setTimeout(function (){
        redraw();
    }, 200);

    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        url: "/randomize",
        dataType: "json",
        success: function (result) {
            sliceImage.src = 'data:image/jpeg;base64,'+result['imageData'].split('\'')[1];
            redraw();
        }
    });

    setTimeout(function (){
        redraw();
    }, 200);
    updatePlot()
}
