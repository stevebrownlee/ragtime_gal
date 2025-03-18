# template.py
# Contains HTML templates for the Flask application

UPLOAD_FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PDF & Markdown Embedding Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .form-container, .query-container, .admin-container {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .admin-container {
            background-color: #fff0f0;
        }
        .dropzone {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 25px;
            text-align: center;
            margin-bottom: 15px;
            transition: all 0.3s ease;
            background-color: #f8f9fa;
            cursor: pointer;
        }
        .dropzone.dragover {
            border-color: #4CAF50;
            background-color: #e8f5e9;
        }
        .file-list {
            margin-top: 15px;
            max-height: 150px;
            overflow-y: auto;
            border: 1px solid #eee;
            padding: 10px;
            border-radius: 4px;
            background-color: white;
        }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .query-options {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        .query-option {
            flex: 1;
        }
        select, input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        select:focus, input[type="number"]:focus {
            border-color: #4CAF50;
            outline: none;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        input[type="file"] {
            margin-bottom: 15px;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .danger-button {
            background-color: #f44336;
        }
        .danger-button:hover {
            background-color: #d32f2f;
        }
        .response {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f7ef;
            border-left: 4px solid #4CAF50;
            display: none;
        }
    </style>
</head>
<body>
    <h1>PDF & Markdown Embedding Tool</h1>

    <div class="query-container">
        <h2>Query Embedded Documents</h2>
        <form id="queryForm">
            <label for="queryInput">Enter your question:</label>
            <input type="text" id="queryInput" name="query" placeholder="What would you like to know about the document?" required>

            <div class="query-options">
                <div class="query-option">
                    <label for="templateSelect">Response style:</label>
                    <select id="templateSelect" name="template">
                        <option value="standard">Standard</option>
                        <option value="creative">Creative</option>
                        <option value="sixthwood" selected>Sixth Wood</option>
                    </select>
                </div>

                <div class="query-option">
                    <label for="temperatureInput">Temperature (0-2):</label>
                    <input type="number" id="temperatureInput" name="temperature" min="0" max="2" step="0.1" value="1.0">
                </div>
            </div>

            <button type="submit">Submit Query</button>
        </form>
        <div id="queryResponse" class="response"></div>
    </div>

    <div class="form-container">
        <h2>Upload Documents</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div id="dropzone" class="dropzone">
                <p>Drag and drop PDF or Markdown files here, or click to select files</p>
                <input type="file" id="pdfFile" name="file" accept=".pdf,.md" multiple style="display: none;">
            </div>
            <div id="fileList" class="file-list" style="display: none;">
                <h3>Selected Files:</h3>
                <div id="fileItems"></div>
            </div>
            <button type="submit" id="uploadButton" disabled>Upload and Embed</button>
        </form>
        <div id="uploadResponse" class="response"></div>
    </div>

    <div class="admin-container">
        <h2>Database Administration</h2>
        <p><strong>Warning:</strong> The following actions are irreversible.</p>
        <button id="purgeButton" class="danger-button">Purge Database</button>
        <div id="purgeResponse" class="response"></div>
    </div>

    <script>
        // File upload handling with drag and drop
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('pdfFile');
        const fileList = document.getElementById('fileList');
        const fileItems = document.getElementById('fileItems');
        const uploadButton = document.getElementById('uploadButton');
        const uploadForm = document.getElementById('uploadForm');
        const uploadResponse = document.getElementById('uploadResponse');

        // Track selected files
        let selectedFiles = [];

        // Handle drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropzone.classList.add('dragover');
        }

        function unhighlight() {
            dropzone.classList.remove('dragover');
        }

        // Handle dropped files
        dropzone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

        // Handle files selected via the file input
        dropzone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
        });

        function handleFiles(files) {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                // Check if file is PDF or Markdown
                if (file.type === 'application/pdf' ||
                    file.name.toLowerCase().endsWith('.md') ||
                    file.type === 'text/markdown') {
                    addFile(file);
                }
            }
            updateFileList();
        }

        function addFile(file) {
            // Check if file already exists in the selection
            if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
                selectedFiles.push(file);
            }
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateFileList();
        }

        function updateFileList() {
            if (selectedFiles.length > 0) {
                fileList.style.display = 'block';
                uploadButton.disabled = false;

                fileItems.innerHTML = '';
                selectedFiles.forEach((file, index) => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';

                    const fileName = document.createElement('div');
                    fileName.textContent = file.name;

                    const removeBtn = document.createElement('span');
                    removeBtn.className = 'remove-file';
                    removeBtn.textContent = '×';
                    removeBtn.addEventListener('click', () => removeFile(index));

                    fileItem.appendChild(fileName);
                    fileItem.appendChild(removeBtn);
                    fileItems.appendChild(fileItem);
                });
            } else {
                fileList.style.display = 'none';
                uploadButton.disabled = true;
            }
        }

        // Handle form submission for multiple files
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            if (selectedFiles.length === 0) {
                return;
            }

            uploadResponse.style.display = 'block';
            uploadResponse.textContent = 'Uploading and embedding files...';
            uploadButton.disabled = true;

            let successCount = 0;
            let errorCount = 0;
            let results = [];

            for (let i = 0; i < selectedFiles.length; i++) {
                const file = selectedFiles[i];
                const formData = new FormData();
                formData.append('file', file);

                try {
                    uploadResponse.textContent = `Processing file ${i+1} of ${selectedFiles.length}: ${file.name}`;

                    const response = await fetch('/embed', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        successCount++;
                        results.push(`✓ ${file.name}: ${result.message}`);
                    } else {
                        errorCount++;
                        results.push(`✗ ${file.name}: ${result.error}`);
                    }
                } catch (error) {
                    errorCount++;
                    results.push(`✗ ${file.name}: ${error.message}`);
                }
            }

            uploadResponse.innerHTML = `
                <p><strong>Upload complete:</strong> ${successCount} successful, ${errorCount} failed</p>
                <ul>${results.map(r => `<li>${r}</li>`).join('')}</ul>
            `;

            uploadResponse.style.borderLeft = errorCount === 0 ? '4px solid #4CAF50' : '4px solid #f44336';
            uploadButton.disabled = false;

            // Clear file list after successful upload
            if (errorCount === 0) {
                selectedFiles = [];
                updateFileList();
            }
        });

        // Query form handling
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const queryInput = document.getElementById('queryInput').value;
            const templateSelect = document.getElementById('templateSelect');
            const temperatureInput = document.getElementById('temperatureInput');
            const responseDiv = document.getElementById('queryResponse');

            responseDiv.style.display = 'block';
            responseDiv.textContent = 'Processing query...';

            // Prepare request data
            const requestData = {
                query: queryInput
            };

            // Add template if selected
            if (templateSelect && templateSelect.value) {
                requestData.template = templateSelect.value;
            }

            // Add temperature if provided
            if (temperatureInput && temperatureInput.value) {
                requestData.temperature = parseFloat(temperatureInput.value);
            }

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });

                const result = await response.json();
                responseDiv.textContent = result.message || result.error;
                responseDiv.style.borderLeft = response.ok ? '4px solid #4CAF50' : '4px solid #f44336';
            } catch (error) {
                responseDiv.textContent = 'Error: ' + error.message;
                responseDiv.style.borderLeft = '4px solid #f44336';
            }
        });

        // Purge database handling
        document.getElementById('purgeButton').addEventListener('click', async function() {
            if (confirm('Are you sure you want to purge all documents from the database? This action cannot be undone.')) {
                const responseDiv = document.getElementById('purgeResponse');

                responseDiv.style.display = 'block';
                responseDiv.textContent = 'Purging database...';

                try {
                    const response = await fetch('/purge', {
                        method: 'POST'
                    });

                    const result = await response.json();
                    responseDiv.textContent = result.message || result.error;
                    responseDiv.style.borderLeft = response.ok ? '4px solid #4CAF50' : '4px solid #f44336';
                } catch (error) {
                    responseDiv.textContent = 'Error: ' + error.message;
                    responseDiv.style.borderLeft = '4px solid #f44336';
                }
            }
        });
    </script>
</body>
</html>
"""