// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsDisplay = document.getElementById('resultsDisplay');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadBtn = document.getElementById('downloadBtn');
const printBtn = document.getElementById('printBtn');
const previewImage = document.getElementById('previewImage');

// State
let currentFile = null;
let currentResults = null;

// Event Listeners
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleFileDrop);
newAnalysisBtn.addEventListener('click', resetUI);
downloadBtn.addEventListener('click', downloadReport);
printBtn.addEventListener('click', printReport);

// File Upload Handlers
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file
    const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'application/dicom'];
    const maxSize = 50 * 1024 * 1024; // 50 MB
    
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm')) {
        showError('Invalid file type. Please upload JPG, PNG, TIFF, or DICOM files.');
        return;
    }
    
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 50MB.');
        return;
    }
    
    currentFile = file;
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show preview and loading state
    showLoading(file);
    
    // Upload file
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentResults = data;
            displayResults(data);
            hideLoading();
        } else {
            showError(data.error || 'Analysis failed');
            hideLoading();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Network error. Please try again.');
        hideLoading();
    });
}

function showLoading(file) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsDisplay.style.display = 'none';
    loadingIndicator.style.display = 'block';
    errorMessage.style.display = 'none';
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
}

function displayResults(data) {
    const hasHemorrhage = data.has_hemorrhage;
    const confidence = data.confidence;
    const riskLevel = data.risk_level;
    
    // Update result value
    const resultValue = document.getElementById('resultValue');
    resultValue.textContent = data.result;
    resultValue.className = 'result-value ' + (hasHemorrhage ? 'hemorrhage' : 'no-hemorrhage');
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = confidence + '%';
        confidenceText.textContent = confidence + '%';
    }, 100);
    
    // Update risk badge
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = riskLevel;
    riskBadge.className = 'risk-badge ' + getRiskClass(riskLevel);
    
    // Update file info
    document.getElementById('filenameInfo').textContent = data.filename || 'Unknown';
    document.getElementById('timestampInfo').textContent = new Date(data.upload_time).toLocaleString();
    document.getElementById('statusInfo').textContent = 'Analysis Complete - ' + new Date().toLocaleTimeString();
    
    // Show results display
    resultsDisplay.style.display = 'flex';
}

function getRiskClass(riskLevel) {
    switch(riskLevel) {
        case 'Low Risk':
            return 'low';
        case 'Medium Risk':
            return 'medium';
        case 'High Risk':
            return 'high';
        default:
            return 'low';
    }
}

function showError(message) {
    resultsSection.style.display = 'block';
    resultsDisplay.style.display = 'none';
    loadingIndicator.style.display = 'none';
    errorMessage.style.display = 'block';
    errorMessage.textContent = '❌ ' + message;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetUI() {
    resultsSection.style.display = 'none';
    resultsDisplay.style.display = 'none';
    loadingIndicator.style.display = 'none';
    errorMessage.style.display = 'none';
    fileInput.value = '';
    currentFile = null;
    currentResults = null;
    previewImage.src = '';
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadReport() {
    if (!currentResults) return;
    
    const report = generateReport(currentResults);
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(report));
    element.setAttribute('download', `hemorrhage_report_${Date.now()}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

function printReport() {
    if (!currentResults) return;
    
    const report = generateReport(currentResults);
    const printWindow = window.open('', '', 'height=500,width=800');
    printWindow.document.write('<pre>' + escapeHtml(report) + '</pre>');
    printWindow.document.close();
    printWindow.print();
}

function generateReport(results) {
    const timestamp = new Date().toLocaleString();
    const report = `
================================================================================
BRAIN HEMORRHAGE DETECTION SYSTEM - ANALYSIS REPORT
================================================================================

Date & Time: ${timestamp}
File Name: ${results.filename || 'Unknown'}

================================================================================
ANALYSIS RESULTS
================================================================================

Detection Result: ${results.result}
Confidence Level: ${results.confidence}%
Risk Assessment: ${results.risk_level}

================================================================================
INTERPRETATION
================================================================================

Result: ${results.has_hemorrhage ? 'HEMORRHAGE DETECTED' : 'NO HEMORRHAGE DETECTED'}
Confidence: ${results.confidence}% (${results.confidence > 75 ? 'HIGH' : results.confidence > 50 ? 'MEDIUM' : 'LOW'} CONFIDENCE)

================================================================================
IMPORTANT DISCLAIMER
================================================================================

This analysis is provided by an AI system for assistive purposes only.
This tool is NOT a substitute for professional medical diagnosis and should
not be used for clinical decision-making without consultation with a qualified
medical professional.

Always consult with a licensed physician for proper medical diagnosis and
treatment planning.

================================================================================
System Information
================================================================================

Application: Brain Hemorrhage Detection System v1.0
AI Model: Convolutional Neural Network (CNN)
Accuracy: 96.5%

================================================================================

Generated by Brain Hemorrhage Detection System
All rights reserved.
`;
    return report;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Initialize
console.log('Application initialized successfully');

// Log app info on load
fetch('/api/info')
    .then(response => response.json())
    .then(data => {
        console.log('App Info:', data);
    })
    .catch(error => console.log('Could not fetch app info:', error));
