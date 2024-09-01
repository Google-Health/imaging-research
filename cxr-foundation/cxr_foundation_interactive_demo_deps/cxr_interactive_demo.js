/**
 * @fileoverview This file contains the JavaScript code necessary for the
 * operation of the CXR Interactive Demo in a Colab environment.
 *
 * Ensure that the access token is securely added to the global scope before
 * invoking any API functions. To set the access token, include the following
 * script tag in your HTML:
 *
 * <script>
 *     let accessToken = 'YOUR_ACCESS_TOKEN_HERE';
 * </script>
 */

let error = '';
const terminologyTags = ['absent', 'present'];
const defaultTerminologyTags = ['absent', 'present'];
const gtCssClassMatchingTermonology = ['negative', 'positive'];
const aiCssClassMatchingTermonology = ['ai-negative', 'ai-positive'];

let threshold = 0.5;
let trainingSetPct = 0.5;

// State
let embeddingsArr = [];
let base64PngArr = [];  // To hold the actual pixel bytes
let trainingIndices = [];
let evaluationIndices = [];
let sortedRocData = [];
let model = undefined;
let resultsArr = [];  // {fileName, label, viewerEl, score}

const viewersContrainerEl = document.getElementById('dicomImage');

/**
 * Clear all state
 */
function clearDicoms() {
  viewersContrainerEl.innerHTML = '';
  embeddingsArr = [];
  base64PngArr = [];
  trainingIndices = [];
  evaluationIndices = [];
  sortedRocData = [];
  model = undefined;
  resultsArr = [];
  document.getElementById('step2').style.display = 'none';
  document.getElementById('step3').style.display = 'none';
  document.getElementById('embeddingsButton').style.display = '';
}

function updateTerminology(el) {
  const label = Number(el.name);
  const value = el.value || defaultTerminologyTags[label];
  terminologyTags[label] = value;
  for (const badgeEl of document.querySelectorAll(
           `.${gtCssClassMatchingTermonology[label]} .badge.gt`)) {
    badgeEl.innerText = value;
  }
  for (const badgeEl of document.querySelectorAll(
           `.${aiCssClassMatchingTermonology[label]} .badge.ai`)) {
    badgeEl.innerText = value;
  }
}

function updateTextBasedOnTermonology(
    textEl, terminologyTags, labelEl, cssClassTermonology) {
  const label = labelEl.classList.contains(cssClassTermonology[1]) ? 1 : 0;
  textEl.innerText = terminologyTags[label];
}

// Call the function to fetch and process labels

/**
 * @fileoverview OAuth token for codeplay client.
 */

/**
 * Load Google Identity Services Library if not already loaded.
 */
function loadGoogleIdentityServicesLibrary() {
  if (typeof google !== 'undefined' && google.accounts && google.accounts.oauth2) {
    return; // Library is already loaded
  }

  // Create a script element to load the Google Identity Services script
  const script = document.createElement('script');
  script.src = 'https://accounts.google.com/gsi/client';
  document.head.appendChild(script); // Append the script to the document head
}

loadGoogleIdentityServicesLibrary();

/**
 * Get access token for the user.
 * @return {!Promise<string>}
 */
function getAccessToken() {
  // Import oauth2 library from google, if not already loaded.

  return new Promise((resolve, reject) => {
    console.log('Initializing OAuth client');

    const client = google.accounts.oauth2.initTokenClient({
      client_id: '75641001687-e4idtek5ba63psb2lpi60fdp7dfl9s4u.apps.googleusercontent.com',
      scope: "https://www.googleapis.com/auth/cloud-platform",
      callback: (tokenResponse) => {
        if (tokenResponse && tokenResponse.access_token) {
          console.log('OAuth initialized');
          resolve(tokenResponse.access_token);
        } else {
          reject(new Error('Login failed'));
        }
      },
    });

    client.callback = (res) => {
      if (res.error) {
        console.error('Error during OAuth callback:', res);
        reject(res.error);
      } else {
        resolve(res.access_token);
      }
    };

    client.requestAccessToken();
  });
}

/////////// Cornerstone viewer
try {
  window.cornerstoneWADOImageLoader.webWorkerManager.initialize({
    maxWebWorkers: 4,
    startWebWorkersOnDemand: true,
    webWorkerTaskPaths: [],
    taskConfiguration: {
      decodeTask: {
        initializeCodecsOnStartup: true,
        strict: true,
      },
    },
  });
} catch (error) {
  throw new Error('cornerstoneWADOImageLoader is not loaded');
}

cornerstoneWADOImageLoader.external.cornerstone = cornerstone;

function posNegBadgeToggler(textEl, terminologyTags, labelEl, cssClassTermonology) {
  labelEl.classList.toggle('positive');
  labelEl.classList.toggle('negative');
  updateTextBasedOnTermonology(
      textEl, terminologyTags, labelEl, gtCssClassMatchingTermonology);
}

// Function to create a new viewer element
function createViewerElement(index, name, label) {

  const defaultLabel = gtCssClassMatchingTermonology[label];
  const containerDiv = document.createElement('div');
  containerDiv.className = 'viewer-container ' + defaultLabel;
  const viewerDiv = document.createElement('div');
  viewerDiv.id = `dicomImage${index}`;
  viewerDiv.className = 'viewer';
  containerDiv.appendChild(viewerDiv);
  const zoomDiv = document.createElement('div');
  zoomDiv.addEventListener('click', () => {
    viewerDiv.classList.toggle('popup');
    viewersContrainerEl.classList.toggle('childPopup');
  });
  zoomDiv.className = 'zoomButton';
  zoomDiv.innerText = '⛶';
  viewerDiv.appendChild(zoomDiv);

  const gtDiv = document.createElement('div');
  gtDiv.className = 'badge gt';
  gtDiv.addEventListener(
      'click',
      () => posNegBadgeToggler(
          gtDiv, terminologyTags, containerDiv, gtCssClassMatchingTermonology));
  updateTextBasedOnTermonology(
      gtDiv, terminologyTags, containerDiv, gtCssClassMatchingTermonology);
  containerDiv.appendChild(gtDiv);

  const aiDiv = document.createElement('div');
  aiDiv.className = 'badge ai';
  containerDiv.appendChild(aiDiv);

  const buttonDiv = document.createElement('div');
  buttonDiv.innerText = '🔄';
  buttonDiv.className = 'toggleButton';
  buttonDiv.addEventListener(
      'click',
      () => posNegBadgeToggler(
          gtDiv, terminologyTags, containerDiv, gtCssClassMatchingTermonology));
  containerDiv.appendChild(buttonDiv);
  const span = document.createElement('div');
  span.className = 'badge dataset training';
  containerDiv.appendChild(span);
  containerDiv.fileName = name;
  containerDiv.title = name + "\\nWindow with left mouse button. Pan with middle. Zoom with right.";
  viewersContrainerEl.appendChild(containerDiv);
  cornerstone.enable(viewerDiv);
  return viewerDiv;
}

function loadFileList(files) {
  waiting(false);
  const label =
  document.getElementById('defaultLabel').classList.contains('positive') ?
  1 :
  0;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);

    // Create a new viewer if needed
    const viewerElement = createViewerElement(i, file.name, label);

    displayImageIdInElement(imageId, viewerElement);
    saveBase64PNG(imageId);
  }
}

function saveBase64PNG(imageId) {
  cornerstone.loadImage(imageId).then(
      function(image) {
        // Extract pixel data
        const pixelData = image.getPixelData();
        const width = image.width;
        const height = image.height;

        // Create a canvas and draw the image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        const imageData = context.createImageData(width, height);
        for (let i = 0; i < pixelData.length; i++) {
          imageData.data[i * 4] = pixelData[i];      // Red
          imageData.data[i * 4 + 1] = pixelData[i];  // Green
          imageData.data[i * 4 + 2] = pixelData[i];  // Blue
          imageData.data[i * 4 + 3] = 255;           // Alpha
        }
        context.putImageData(imageData, 0, 0);

        // Convert canvas to base64 PNG
        const base64Png = canvas.toDataURL('image/png');
        base64PngArr.push(base64Png);
      },
      function(err) {
        alert(err);
      });
}

function displayImageIdInElement(imageId, targetEl) {
  cornerstone.loadImage(imageId).then(
      function(image) {
        const viewport =
            cornerstone.getDefaultViewportForImage(targetEl, image);
        cornerstone.displayImage(targetEl, image, viewport);
        cornerstone.resize(targetEl, true);
        cornerstoneTools.mouseInput.enable(targetEl);
        // cornerstoneTools.mouseWheelInput.enable(element);
        cornerstoneTools.wwwc.activate(
            targetEl, 1);  // ww/wc is the default tool for left mouse button
        cornerstoneTools.pan.activate(
            targetEl, 2);  // pan is the default tool for middle mouse button
        cornerstoneTools.zoom.activate(
            targetEl, 4);  // zoom is the default tool for right mouse button
        // cornerstoneTools.zoomWheel.activate(element); // zoom is the
        // default tool for middle mouse wheel
      },
      function(err) {
        alert(err);
      });
}

document.getElementById('selectFile').addEventListener('change', function(e) {
  const files = e.target.files;
  loadFileList(files);
});

//////// Make Step 2 appear
const step2Div = document.getElementById('step2');
const selectFile = document.getElementById('selectFile');
selectFile.addEventListener('change', () => {
  if (selectFile.files.length > 0) {
    step2Div.style.display = 'block';
    document.getElementById('step3').style.display = 'none';
    document.getElementById('embeddingsButton').style.display = '';
  }
});

//////// CSV
function downloadCSV() {
  let csv = '';
  var children = document.getElementById('dicomImage').children;
  for (var i = 0; i < children.length; i++) {
    const c = children[i];
    csv += c.fileName + ',' + c.classList.contains('positive') + '\\n';
  }
  downloadStringAsFile('labes.csv', csv);
}

const commentEl = document.getElementById('commentEl');

////// Compute Embeddings
function floatsToBase64(floatArray) {
  // Create a buffer from the float array (using Float32Array for this example)
  let buffer = new Float32Array(floatArray).buffer;

  // Convert the ArrayBuffer to a binary string
  let binaryString = Array.from(new Uint8Array(buffer))
                         .map(byte => String.fromCharCode(byte))
                         .join('');

  // Convert the binary string to a Base64 string
  return btoa(binaryString);
}

function base64ToFloats(base64String) {
  // Decode the Base64 string to a binary string
  let binaryString = atob(base64String);
  let charList = binaryString.split('');
  let uintArray = new Uint8Array(charList.map(char => char.charCodeAt(0)));

  // Convert the binary data back to a float array (using Float32Array for this
  // example)
  return new Float32Array(uintArray.buffer);
}

async function computeEmbeddings() {
  const useCache = document.getElementById('cacheToggle').checked;
  commentEl.innerText = '';
  const children = document.getElementById('dicomImage').children;
  try {
    if (!accessToken) {
      accessToken = await getAccessToken();
    }
    waiting(true);
    for (let i = embeddingsArr.length; i < base64PngArr.length; i++) {
      commentEl.innerText = 'Computing embedding ' + (i + 1) + ' out of ' +
          base64PngArr.length + ',   ';
      const img = base64PngArr[i];
      const el = children[i];
      el.classList.add('loading');
      el.scrollIntoView({behavior: 'smooth', block: 'start'});
      let embeddings = undefined;
      if (useCache && el.fileName) {
        const stored = localStorage.getItem(el.fileName);
        if (stored) {
          embeddings = base64ToFloats(stored);
        }
      }
      if (!embeddings) {
        let tries = 3;
        while (!embeddings && tries > 0) {
          try {
            embeddings = await callVertexAIEndpoint(img, accessToken);
          } catch (error) {
            tries = tries - 1;
            if (!tries) {
              throw error;
            }
          }
        }
        embeddings = embeddings.predictions[0][0];
        if (useCache && el.fileName) {
          localStorage.setItem(el.fileName, floatsToBase64(embeddings));
        }
      }
      el.classList.add('embedding');
      el.classList.remove('loading');
      embeddingsArr.push(embeddings);
    }
  } catch (error) {
    console.error('Error during computeEmbeddings:', {error});
    showErrorToast('Error during computeEmbeddings: ' + error.message);
    commentEl.innerText =
        'Error during computing embeddings, click button again to retry. [' +
        error.message +
        ']. Make sure you were approved for access. See top of page.';
    waiting(false);
    return;
  }
  waiting(false);
  const step3El = document.getElementById('step3');
  step3El.style.display = 'block';
  step3El.scrollIntoView({behavior: 'smooth', block: 'start'});
  document.getElementById('embeddingsButton').style.display = 'none';
  updateTrainingSet();
}

//////// TRAINING
async function trainModel() {
  updateTrainingSet();
  const children = document.getElementById('dicomImage').children;
  var labels = trainingIndices.map(i => {
    return children[i].classList.contains('positive') ? 1 : 0;
  });
  if (!labels.filter(label => label === 1).length) {
    showErrorToast('Training set has no "' + terminologyTags[1] + '" samples.');
    return;
  }
  if (!labels.filter(label => label === 0).length) {
    showErrorToast('Training set has no "' + terminologyTags[0] + '" samples.');
    return;
  }
  waiting(true);
  setTimeout(async () => {
    // 1. Load and Prepare Data
    const xs = tf.tensor(trainingIndices.map(i => embeddingsArr[i]));


    const ys = tf.tensor(labels);

    // 2. Define the Model Architecture
    model = tf.sequential();
    model.add(tf.layers.dense(
        {units: 512, activation: 'relu', inputShape: xs.shape[1]}));
    model.add(tf.layers.dense({units: 256, activation: 'relu'}));
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));  // Output layer for binary classification
    // 3. Compile the Model
    model.compile(
        {optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});

    // 4. Train the Model
    await model.fit(xs, ys, {epochs: 100, batchSize: 100});

    // 5. Make Predictions
    const predictions = model.predict(tf.tensor(embeddingsArr));

    packageResults(await predictions.array());
    let skipEval = false;
    if (!evaluationIndices.map(i => resultsArr[i])
             .filter(r => r.label === 1)
             .length) {
      showErrorToast(
          'Eval set has no "' + terminologyTags[1] +
          '" samples, can not evaluate.');
      skipEval = true;
    }
    if (!evaluationIndices.map(i => resultsArr[i])
             .filter(r => r.label === 0)
             .length) {
      showErrorToast(
          'Eval set has no "' + terminologyTags[0] +
          '" samples, can not evaluate.');
      skipEval = true;
    }

    if (!skipEval) {
      sortedRocData = computeROCData(evaluationIndices.map(i => resultsArr[i]));
      if (sortedRocData) {
        plotROC(sortedRocData);  // stores the rocData

        // update threshold bar
        document.getElementById('threshold').max =
            sortedRocData[sortedRocData.length - 1].threshold + 0.01;
        document.getElementById('threshold').min =
            sortedRocData.at(0).threshold;
      }
      displayPredictions();
    }
    waiting(false);
    document.getElementById('predictionNotes').style.display = '';
    document.getElementById('downloadButton').style.display = '';
  }, 50);
}

function downloadRawResults() {
  let csv = `dataset,filename,label,score\\n`;
  evaluationIndices.map((i) => resultsArr[i]).forEach((r) => {
    csv = csv + `eval,${r.fileName},${r.label},${r.score}\\n`;
  });
  trainingIndices.map((i) => resultsArr[i]).forEach((r) => {
    csv = csv + `training,${r.fileName},${r.label},${r.score}\\n`;
  });

  downloadStringAsFile('predictions.csv', csv);
}

function downloadModel() {
  if (model) {
    model.save('downloads://trained-model');
  }
}


document.getElementById('threshold').oninput =
    function() {
  threshold = parseFloat(this.value);
  displayPredictions();
}


document.getElementById('trainingSetPct').oninput =
    function() {
  document.getElementById('trainingSetPctShow').innerText = this.value + '%';
  trainingSetPct = this.value / 100;
  updateTrainingSet();
}


function updateTrainingSet() {
  const children = document.getElementById('dicomImage').children;
  // Step 1: Get all children elements and their labels
  const labels = Array.from(children).map(child => {
    return child.classList.contains('positive') ? 1 : 0;
  });

  // Step 2: Separate indices of positive and negative samples
  let positiveIndices = [];
  let negativeIndices = [];
  labels.forEach((label, index) => {
    if (label === 1) {
      positiveIndices.push(index);
    } else {
      negativeIndices.push(index);
    }
  });

  // Step 3: Calculate number of samples for training
  let numPositiveForTraining =
      Math.ceil(positiveIndices.length * trainingSetPct);
  let numNegativeForTraining =
      Math.ceil(negativeIndices.length * trainingSetPct);

  // Step 4: Select indices for training and evaluation
  trainingIndices =
      positiveIndices.slice(0, numPositiveForTraining)
          .concat(negativeIndices.slice(0, numNegativeForTraining));
  evaluationIndices =
      positiveIndices.slice(numPositiveForTraining)
          .concat(negativeIndices.slice(numNegativeForTraining));

  for (const i of trainingIndices) {
    children[i].classList.add('training');
    children[i].classList.remove('eval');
  }
  for (const i of evaluationIndices) {
    children[i].classList.add('eval');
    children[i].classList.remove('training');
  }
}

function packageResults(predictionArray) {
  resultsArr = [];
  const children = document.getElementById('dicomImage').children;
  predictionArray.forEach((predScore, index) => {
    const viewerEl = children.item(index);
    const fileName = viewerEl.fileName;
    const label =
        viewerEl.className.includes('positive') ? 1 : 0;  // Store as 0 or 1
    resultsArr.push({
      fileName: viewerEl.fileName,
      label: label,
      viewerEl: viewerEl,
      score: predScore[0],
    });
  });
}

/**
 * Display predictions based on the threshold.
 */
function displayPredictions() {
  resultsArr.forEach((result) => {
    const el = result.viewerEl;
    if (result.score >= threshold) {
      el.classList.add('ai-positive');
      el.classList.remove('ai-negative');
      updateTextBasedOnTermonology(
          el.querySelector('.ai'), terminologyTags, el,
          aiCssClassMatchingTermonology);
    } else {
      el.classList.remove('ai-positive');
      el.classList.add('ai-negative');
      updateTextBasedOnTermonology(
          el.querySelector('.ai'), terminologyTags, el,
          aiCssClassMatchingTermonology);
    }
  });
  const auc = computeAUC(sortedRocData);
  const rocDataPoint = findClosestRocData(sortedRocData, threshold);
  highlightPoint(rocDataPoint);
  const tpr = rocDataPoint.y;
  const fpr = rocDataPoint.x;
  const resultsDisplay = document.getElementById('metrics');

  // Clear previous results
  resultsDisplay.innerHTML = '';

  // Create HTML content for the results
  const content = `
        <span style="font-weight: 900;">Metrics:</span>
        <p>Threshold: ${threshold.toFixed(2)}</p>
        <p>AUC: ${auc.toFixed(2)}</p>
        <p>TPR: ${tpr.toFixed(2)}</p>
        <p>FPR: ${fpr.toFixed(2)}</p>
    `;

  // Display results
  resultsDisplay.innerHTML = content;
}

let rocChart;  // Define rocChart in a higher scope to maintain its reference

function plotROC(rocData) {
  const ctx = document.getElementById('rocChart').getContext('2d');

  // Destroy the existing chart if it exists
  if (rocChart) {
    rocChart.destroy();
  }

  // Create a new chart instance
  rocChart = new Chart(ctx, {
    data: {
      datasets: [
        {
          type: 'line',
          label: 'ROC Curve',
          data: rocData,
          fill: false,
          borderColor: 'rgb(255, 99, 132)',
          tension: 0,
        },
        {
          type: 'scatter',
          data: [rocData[0]],
          fill: false,
          borderColor: 'rgb(255, 99, 132)',
          pointRadius: 6,
          pointBackgroundColor: 'yellow',
          pointBorderColor: 'yellow',
          tension: 0,
        }
      ]
    },
    options: {
      maintainAspectRatio: true,
      aspectRatio: 1,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {display: true, text: 'False Positive Rate (1 - Specificity)'},
          min: 0,
          max: 1
        },
        y: {
          title: {display: true, text: 'True Positive Rate (Sensitivity)'},
          min: 0,
          max: 1
        }
      },
      plugins: {
        legend: {
          display: false,
        }
      },
    },
  });
  document.getElementById('rocChart').style.display = '';
}

function highlightPoint(rocDataPoint) {
  if (!rocChart) {
    return;
  }
  rocChart.data.datasets[1].data = [rocDataPoint];
  rocChart.update();
}

///// Generate JS/HTML util functions.
function showErrorToast(message) {
  // Create the toast container if it doesn't exist
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  // Create the toast element
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;

  // Append the toast to the container
  toastContainer.appendChild(toast);

  // Trigger the fade-in effect
  setTimeout(() => {
    toast.style.opacity = '1';
  }, 10);

  // Remove the toast after the specified duration
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => {
      toastContainer.removeChild(toast);
    }, 500);
  }, 3000);
}

function waiting(iswaiting) {
  spinner(iswaiting);
  Array.from(document.getElementsByTagName('button')).forEach((b) => {
    b.disabled = iswaiting;
  });
  document.getElementById('selectFile').disabled = iswaiting;
}

function spinner(enabled) {
  document.getElementById('spinner').className = enabled ? 'loader' : '';
}
/**
 * @fileoverview This file contains the code to load the CXR14 dataset from
 * Google Cloud Storage.
 */

const GCS_BUCKET_NAME = 'cxr-foundation-demo';

/** Populates the diagnosis combo box with the CXR14 diagnosis list. */
function populateDiagnosisCXR14Combo() {
  const DIAGNOSIS_CXR14 = [
    'AIRSPACE_OPACITY', 'ATELECTASIS', 'CARDIOMEGALY', 'CONSOLIDATION',
    'EFFUSION', 'FRACTURE', 'PNEUMOTHORAX', 'PULMONARY_EDEMA'
  ];
  let select = document.getElementById('diagnosis-cxr14');
  for (let i = 0; i < DIAGNOSIS_CXR14.length; i++) {
    const opt = DIAGNOSIS_CXR14[i];
    const el = document.createElement('option');
    el.text = opt;
    el.value = opt;

    select.add(el);
  }
}

populateDiagnosisCXR14Combo();

/**
 * Fetches the CXR14 dataset from Google Cloud Storage.
 * @param {string} diagnosis The diagnosis to filter the dataset by.
 * @param {number} limit The maximum number of images to load.
 */
async function fetchCXR14Dataset(diagnosis, limit) {
  if (!accessToken) {
    accessToken = await getAccessToken();
  }
  const fileName = 'cxr14/labels.csv';

  try {
    // Construct the URL for the Google Cloud Storage API
    const url = `https://storage.googleapis.com/storage/v1/b/${
        GCS_BUCKET_NAME}/o/${encodeURIComponent(fileName)}?alt=media`;

    // Fetch the CSV file
    const response = await fetch(
        url,
        {headers: new Headers({'Authorization': `Bearer ${accessToken}`})});

    if (!response.ok) {
      const bodyText = await response.text();
      console.log('FetchCXR14Dataset error: ', {response});
      showErrorToast(
          'Failed to fetch the CSV file: ' + (response.statusText || bodyText));
      return;
    }

    const csvText = await response.text();

    // Process the CSV text
    const labels = processCsv(csvText)
                       .filter(label => label[diagnosis] !== '')
                       .map(label => {
                         const remoteDicomFile = 'cxr14/inputs/' +
                             label.image_id.replace('.png', '.dcm');
                         return {
                           value: (parseFloat(label[diagnosis]) === 1 ? 1 : 0),
                           remoteDicomFile
                         };
                       });

    const negPosRemoteDicomFiles = [[], []];
    for (let i = 0; i < labels.length; ++i) {
      const label = labels[i];
      if (negPosRemoteDicomFiles[label.value].length < limit) {
        negPosRemoteDicomFiles[label.value].push(label.remoteDicomFile);
      }
    }

    await loadGcsFileList(negPosRemoteDicomFiles[0], 0);
    await loadGcsFileList(negPosRemoteDicomFiles[1], 1);
    const step2Div = document.getElementById('step2');
    step2Div.style.display = 'block';
  } catch (error) {
    console.error('Error:', error);
    showErrorToast(error);
  }
}

/**
 * Function to process CSV text and convert to JSON
 * @param {string} csvText The CSV text to process.
 * @return {!Array<!Object>} The processed CSV text as an array of objects.
 */
function processCsv(csvText) {
  const lines = csvText.split('\\n');
  const headers = lines[0].split(',');

  return lines.slice(1).map(line => {
    const data = line.split(',');
    return headers.reduce((obj, nextKey, index) => {
      obj[nextKey] = data[index];
      return obj;
    }, {});
  });
}

/**
 * Function to load a list of files from Google Cloud Storage
 * @param {!Array<string>} files The list of files to load.
 * @param {number} label The label of the files to load.
 */
async function loadGcsFileList(files, label) {
  try {
    waiting(false);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      // 1. Get DICOM data from GCS
      const blob = await fetchDicomBlob(file);
      if (!blob) continue;  // Skip this file if fetching failed

      // 2. Validate if the Blob is a valid DICOM file
      const isValidDicom = await validateDicomBlob(blob);
      if (!isValidDicom) {
        console.warn(`File ${file} is not a valid DICOM file.`);
        continue;  // Skip this file if it's not a valid DICOM
      }

      // 3. Create an imageId from the Blob (Cornerstone requirement)
      const imageId = createImageIdFromBlob(blob, file);
      if (!imageId) continue;  // Skip this file if creating imageId failed

      // 4. Load DICOM into Cornerstone
      await cornerstone.loadImage(imageId);

      // 5. Create a new viewer element
      const viewerElement = createViewerElement(i, file, label);

      // 6. Display and save the image
      displayImageIdInElement(imageId, viewerElement);
      saveBase64PNG(imageId);
    }
  } catch (error) {
    console.error('Error in loadGcsFileList:', error);
  }
}

/**
 * Function to fetch a DICOM file from Google Cloud Storage
 * @param {string} file The file to fetch.
 * @return {!Blob} The fetched DICOM file.
 */
async function fetchDicomBlob(file) {
  try {
    const response = await fetch(
        `https://storage.googleapis.com/storage/v1/b/${GCS_BUCKET_NAME}/o/${
            encodeURIComponent(file)}?alt=media`,
        {headers: new Headers({'Authorization': `Bearer ${accessToken}`})});

    if (!response.ok) {
      console.error(`Failed to fetch file: ${file}`, response.statusText);
      return null;
    }

    return await response.blob();
  } catch (error) {
    console.error(`Error fetching DICOM file ${file}:`, error);
    return null;
  }
}

/**
 * Function to validate if a Blob is a valid DICOM file
 * @param {!Blob} blob The Blob to validate.
 * @return {boolean} True if the Blob is a valid DICOM file.
 */
async function validateDicomBlob(blob) {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const byteArray = new Uint8Array(arrayBuffer);

    // Check if the file contains the DICM prefix at the correct location
    const dicmPrefix = [0x44, 0x49, 0x43, 0x4D];  // "DICM" in ASCII
    const isDicom = byteArray.slice(128, 132).every(
        (value, index) => value === dicmPrefix[index]);

    return isDicom;
  } catch (error) {
    console.error('Error validating DICOM file:', error);
    return false;
  }
}

/**
 * Function to create an imageId from a Blob
 * @param {!Blob} blob The Blob to create an imageId from.
 * @param {string} fileName The file name of the Blob.
 * @return {string} The created imageId.
 */
function createImageIdFromBlob(blob, fileName) {
  try {
    // Register the blob with Cornerstone File Manager
    const imageId =
        cornerstoneWADOImageLoader.wadouri.fileManager.add(blob, fileName);

    // Return the created imageId
    return imageId;
  } catch (error) {
    console.error('Error creating imageId from blob:', error);
    return null;
  }
}

/**
 * @fileoverview Calls the Vertex AI endpoint with the given image bytes and
 * access token.
 */

/**
 * Loads the protobuf library if needed.
 */
function loadProtobufLibrary() {
  if (typeof protobuf !== 'undefined') {
    return; // Library is already loaded
  }

  // Create a script element to load the Google Identity Services script
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/protobufjs/dist/protobuf.min.js';
  document.head.appendChild(script); // Append the script to the document head
}

loadProtobufLibrary();

/**
 * Calls the Vertex AI endpoint with the given image bytes and access token.
 * @param {string} imageBytes The image bytes to send to the endpoint.
 * @param {string} accessToken The access token to use for authentication.
 * @return {!Promise<!Object>} The response from the endpoint.
 */
async function callVertexAIEndpoint(imageBytes, accessToken) {
  const endpointUrl =
      'https://us-central1-aiplatform.googleapis.com/v1/projects/gh-rad-validation-cxrembd-deid/locations/us-central1/endpoints/8327848403333545984:predict';

  const requestBody = {
    instances: [{
      b64: encodeExampleToBase64(
          await createTensorFlowExample(imageBytes.slice(22)))
    }]
  };

  try {
    const response = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      console.log(response);
      const errorData = await response.json();
      throw new Error(
        response.statusText || errorData?.error.status || errorData?.error.message
          || response.status || JSON.stringify(errorData));
    }

    const responseData = await response.json();
    return responseData;  // Process the response from Vertex AI

  } catch (error) {
    console.error('Error calling Vertex AI endpoint:', {error});
    throw error;  // Handle the error appropriately in your application
  }
}

/**
 * Creates a TensorFlow Example proto from the given image bytes.
 * @param {string} base64String The image bytes to encode.
 * @return {!Promise<!ArrayBuffer>} The TensorFlow Example proto.
 */
async function createTensorFlowExample(base64String) {
  const encodedImage = new Uint8Array(
      atob(base64String).split('').map(char => char.charCodeAt(0)));
  const imageFormat = new TextEncoder().encode('png');

  const protoSchema = `
        syntax = "proto3";

        message Example {
            Features features = 1;
        }

        message Features {
            map<string, Feature> feature = 1;
        }

        message Feature {
            oneof kind {
                BytesList bytes_list = 1;
                FloatList float_list = 2;
                Int64List int64_list = 3;
            }
        }

        message BytesList {
            repeated bytes value = 1;
        }

        message FloatList {
            repeated float value = 1;
        }

        message Int64List {
            repeated int64 value = 1;
        }
    `;

  const root = protobuf.parse(protoSchema).root;
  const Example = root.lookupType('Example');

  const example = {
    features: {
      feature: {
        'image/encoded': {bytesList: {value: [encodedImage]}},
        'image/format': {bytesList: {value: [imageFormat]}}
      }
    }
  };

  const message = Example.create(example);
  const buffer = Example.encode(message).finish();
  return buffer;
}

/**
 * Encodes a buffer to base64.
 * @param {!ArrayBuffer} buffer The buffer to encode.
 * @return {string} The base64-encoded buffer.
 */
function encodeExampleToBase64(buffer) {
  if (!buffer || buffer.length === 0) {
    console.error('Buffer is empty or undefined');  // Debugging
    return '';
  }

  const binaryString =
      Array.from(buffer).map(byte => String.fromCharCode(byte)).join('');
  const base64String = btoa(binaryString);
  return base64String;
}

/**
 * @fileoverview Functions to compute metrics for machine learning models.
 */

/**
 * Finds and returns the ROC data point that is closest to the specified threshold.
 * This function assumes the ROC data points are sorted by threshold.
 * If the threshold is less than the first point, the first point is returned.
 * It returns the point where the threshold is the closest without going over,
 * unless the input threshold is higher than all available thresholds, in which case it logs an error and returns null.
 *
 * @param {!Array<!Object>} sortedRocData - An array of objects representing ROC data points, each with a 'threshold' property.
 * @param {number} threshold - The threshold value to find the closest ROC data point to.
 * @return {Object|null} The closest ROC data point or null if no suitable match is found.
 */
function findClosestRocData(sortedRocData, threshold) {
    // Check if the ROC data array is empty
    if (sortedRocData.length === 0) {
        console.log('Empty ROC data');
        return null;
    }

    // Return the first data point if the specified threshold is less than the lowest threshold in the sorted data
    if (threshold < sortedRocData[0].threshold) {
        return sortedRocData[0];
    }

    // Loop through the sorted ROC data
    for (let i = 0; i < sortedRocData.length; i++) {
        // Return the current data point if its threshold is less than or equal to the input threshold
        // and if it is the last element or if the next element has a threshold greater than the input threshold
        if (sortedRocData[i].threshold <= threshold && (i === sortedRocData.length - 1 || sortedRocData[i + 1].threshold > threshold)) {
            return sortedRocData[i];
        }
    }

    // Log and return null if no threshold closely matches the input threshold
    console.log('No matching threshold found for:', threshold);
    return null;
}



/**
 * Computes the Area Under the Curve (AUC) from ROC data points.
 *
 * @param {!Array<!Object>} rocData An array of objects with x, y, and threshold properties representing ROC points.
 * @return {number} The computed AUC.
 */
function computeAUC(rocData) {
  if (!rocData || rocData.length === 0) return 0; // Return 0 if no data points

  let auc = 0;
  let previousPoint = rocData[0]; // Start with the first point

  // Sum up the areas using the trapezoidal rule
  for (let i = 1; i < rocData.length; i++) {
    const currentPoint = rocData[i];
    // The area of a trapezoid is average of the heights (y values) times the width (difference in x values)
    const heightAverage = (currentPoint.y + previousPoint.y) / 2;
    const width = previousPoint.x - currentPoint.x;  // sorted by threshold acending
    auc += heightAverage * width;
    previousPoint = currentPoint;
  }

  return auc;
}

/**
 * Computes the ROC curve for a binary classifier, including thresholds.
 *
 * @param {!Array<Object>} with fields score and label.
 * @return {!Array<!Object>} An array of objects with x, y, and threshold properties.
 */
function computeROCData(scoreLabelArr) {
  scoreLabelArr.sort((a, b) => b.score - a.score); // Sort descending by score
  let tp = 0, fp = 0;
  let rocData = [];
  const totalPositives = scoreLabelArr.filter(data => data.label === 1).length;
  const totalNegatives = scoreLabelArr.length - totalPositives;

  scoreLabelArr.forEach((data, index) => {
    if (data.label === 1) {
      tp++;
    } else {
      fp++;
    }
    const tpr = tp / totalPositives;
    const fpr = fp / totalNegatives;

    // Look ahead to see if the threashold on next point is the same, if so wait with adding point.
    if (index === scoreLabelArr.length -1 || scoreLabelArr[index + 1].score !== data.score) {
      rocData.push({ x: fpr, y: tpr, threshold: data.score });
    }
  });

  rocData.sort((a, b) => a.threshold - b.threshold);
  return rocData;
}
/**
 * @fileoverview Utility functions.
 */

/**
 * Scroll to the bottom of the page.
 */
function scrollDown() {
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: 'smooth'  // 'auto' for instant scrolling
  });
}

/**
 * Remove all children of an element.
 * @param {!Element} el
 */
function removeAllChildren(el) {
  while (el.firstChild) {
    el.removeChild(el.lastChild);
  }
}

/**
 * Download a string as a file.
 * @param {string} filename
 * @param {string} text
 */
function downloadStringAsFile(filename, text) {
  const element = document.createElement('a');
  element.setAttribute(
      'href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}
