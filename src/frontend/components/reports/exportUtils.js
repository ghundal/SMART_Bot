/**
 * Utility functions for exporting reports as CSV and PDF
 */
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

/**
 * Convert data to CSV format and trigger download
 * @param {Object} data - The data to export
 * @param {string} filename - The filename for the download
 */
export function exportToCSV(data, filename = 'smart-report.csv') {
  if (!data) {
    return;
  }
  // Prepare data based on type
  let csvContent = '';
  const csvData = [];
  // Determine data structure
  if (Array.isArray(data)) {
    // Array of objects
    if (data.length > 0) {
      const headers = Object.keys(data[0]);
      csvData.push(headers);
      // Add data rows
      data.forEach((item) => {
        const row = headers.map((header) => item[header]);
        csvData.push(row);
      });
    }
  } else {
    // Single object
    const headers = Object.keys(data);
    csvData.push(['Metric', 'Value']);
    // Add data rows
    headers.forEach((header) => {
      csvData.push([header, data[header]]);
    });
  }
  // Convert to CSV string
  csvData.forEach((row) => {
    const processedRow = row.map((cell) => {
      // Handle special characters, wrap in quotes if needed
      if (cell === null || cell === undefined) {
        return '';
      }
      const cellStr = String(cell);
      if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
        return `"${cellStr.replace(/"/g, '""')}"`;
      }
      return cellStr;
    });
    csvContent += processedRow.join(',') + '\n';
  });
  // Create download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Wrapper function to export report data in the specified format
 * @param {string} reportType - Type of report being exported
 * @param {Object} data - The data to export
 * @param {string} format - Format to export (csv or pdf)
 */
export function exportReport(reportType, data, format = 'csv') {
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `smart-${reportType}-report-${timestamp}.${format}`;

  if (format === 'csv') {
    exportToCSV(data, filename);
  } else if (format === 'pdf') {
    exportToPDF('reportContent', filename);
  }
}

/**
 * Generate a PDF report from the specified element
 * @param {string} elementId - ID of the element to capture
 * @param {string} filename - Filename for the PDF
 */
export function exportToPDF(elementId, filename = 'smart-report.pdf') {
  // Check if element exists
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with ID '${elementId}' not found`);
    alert(
      'PDF export would be implemented with jsPDF and html2canvas libraries or a backend solution.',
    );
    return;
  }

  html2canvas(element).then((canvas) => {
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const imgProps = pdf.getImageProperties(imgData);
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(filename);
  });
}
