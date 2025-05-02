'use client';

import { useEffect, useRef } from 'react';
import styles from './charts.module.css';

// Simple bar chart component
export function BarChart({ data, xKey, yKey, label, height = 250, color = '#4299e1' }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Get max value for scaling
    const maxValue = Math.max(...data.map((item) => item[yKey]));

    // Clear previous chart if any
    if (chartRef.current) {
      chartRef.current.innerHTML = '';

      // Create bars
      data.forEach((item, index) => {
        const barContainer = document.createElement('div');
        barContainer.className = styles.barContainer;

        const bar = document.createElement('div');
        bar.className = styles.bar;
        bar.style.height = `${Math.min(100, (item[yKey] / maxValue) * 100)}%`;
        bar.style.backgroundColor = color;

        // Create tooltip for value
        bar.setAttribute('data-value', item[yKey]);
        bar.addEventListener('mouseover', (e) => {
          const tooltip = document.createElement('div');
          tooltip.className = styles.tooltip;
          tooltip.textContent = `${label || yKey}: ${item[yKey]}`;
          bar.appendChild(tooltip);
        });

        bar.addEventListener('mouseout', () => {
          const tooltip = bar.querySelector(`.${styles.tooltip}`);
          if (tooltip) tooltip.remove();
        });

        const barLabel = document.createElement('span');
        barLabel.className = styles.barLabel;
        barLabel.textContent =
          typeof item[xKey] === 'string' && item[xKey].includes('T')
            ? new Date(item[xKey]).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
            : item[xKey];

        barContainer.appendChild(bar);
        barContainer.appendChild(barLabel);
        chartRef.current.appendChild(barContainer);
      });
    }
  }, [data, xKey, yKey, label, color]);

  return (
    <div className={styles.chartContainer} style={{ height: `${height}px` }}>
      <div ref={chartRef} className={styles.barChartContainer}></div>
    </div>
  );
}

// Pie chart component
export function PieChart({ data, labelKey, valueKey, colors = [] }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Calculate total for percentages
    const total = data.reduce((sum, item) => sum + item[valueKey], 0);

    // Clear previous chart
    if (chartRef.current) {
      chartRef.current.innerHTML = '';

      // Create SVG element
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 100 100');
      svg.classList.add(styles.pieChart);

      // Calculate segments
      let startAngle = 0;
      const centerX = 50;
      const centerY = 50;
      const radius = 50;

      // Create legend container
      const legendContainer = document.createElement('div');
      legendContainer.className = styles.legendContainer;

      // Create segments
      data.forEach((item, index) => {
        const value = item[valueKey];
        const percentage = (value / total) * 100;
        const angleSize = (percentage / 100) * 360;
        const endAngle = startAngle + angleSize;

        // Calculate SVG arc path
        const startX = centerX + radius * Math.cos(((startAngle - 90) * Math.PI) / 180);
        const startY = centerY + radius * Math.sin(((startAngle - 90) * Math.PI) / 180);
        const endX = centerX + radius * Math.cos(((endAngle - 90) * Math.PI) / 180);
        const endY = centerY + radius * Math.sin(((endAngle - 90) * Math.PI) / 180);

        // Determine if the arc is more than 180 degrees
        const largeArcFlag = angleSize > 180 ? 1 : 0;

        // Create path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute(
          'd',
          `M ${centerX} ${centerY} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY} Z`,
        );

        // Set color
        const color = colors[index % colors.length] || `hsl(${(index * 137.5) % 360}, 70%, 60%)`;
        path.setAttribute('fill', color);

        // Add tooltip
        path.setAttribute('data-label', item[labelKey]);
        path.setAttribute('data-value', value);
        path.setAttribute('data-percentage', `${percentage.toFixed(1)}%`);

        path.addEventListener('mouseover', () => {
          path.setAttribute('stroke', '#fff');
          path.setAttribute('stroke-width', '2');
        });

        path.addEventListener('mouseout', () => {
          path.removeAttribute('stroke');
          path.removeAttribute('stroke-width');
        });

        svg.appendChild(path);

        // Add to legend
        const legendItem = document.createElement('div');
        legendItem.className = styles.legendItem;

        const colorBox = document.createElement('div');
        colorBox.className = styles.colorBox;
        colorBox.style.backgroundColor = color;

        const label = document.createElement('span');
        label.textContent = `${item[labelKey]} (${percentage.toFixed(1)}%)`;

        legendItem.appendChild(colorBox);
        legendItem.appendChild(label);
        legendContainer.appendChild(legendItem);

        startAngle = endAngle;
      });

      chartRef.current.appendChild(svg);
      chartRef.current.appendChild(legendContainer);
    }
  }, [data, labelKey, valueKey, colors]);

  return <div ref={chartRef} className={styles.pieChartContainer}></div>;
}

// Tag cloud component for keywords
export function TagCloud({ data, textKey, valueKey, maxFontSize = 24, minFontSize = 12 }) {
  const cloudRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Get max and min values for scaling
    const maxValue = Math.max(...data.map((item) => item[valueKey]));
    const minValue = Math.min(...data.map((item) => item[valueKey]));
    const range = maxValue - minValue;

    // Clear previous cloud
    if (cloudRef.current) {
      cloudRef.current.innerHTML = '';

      // Create tags
      data.forEach((item) => {
        const tag = document.createElement('span');
        tag.className = styles.tag;
        tag.textContent = item[textKey];

        // Scale font size based on value
        const fontSize =
          range === 0
            ? (maxFontSize + minFontSize) / 2
            : minFontSize + ((item[valueKey] - minValue) / range) * (maxFontSize - minFontSize);

        tag.style.fontSize = `${fontSize}px`;

        // Random color hue
        const hue = Math.floor(Math.random() * 360);
        tag.style.backgroundColor = `hsla(${hue}, 70%, 60%, 0.1)`;
        tag.style.color = `hsl(${hue}, 70%, 40%)`;

        // Add tooltip with count
        tag.setAttribute('data-count', item[valueKey]);

        cloudRef.current.appendChild(tag);
      });
    }
  }, [data, textKey, valueKey, maxFontSize, minFontSize]);

  return <div ref={cloudRef} className={styles.tagCloud}></div>;
}

// Line chart component
export function LineChart({ data, xKey, yKey, label, height = 250, color = '#4299e1' }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Get max value for scaling
    const maxValue = Math.max(...data.map((item) => item[yKey]));

    // Clear previous chart
    if (chartRef.current) {
      chartRef.current.innerHTML = '';

      // Create SVG
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${data.length * 50} 100`);
      svg.classList.add(styles.lineChart);

      // Create line path
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      let pathData = '';

      // Create points for line
      const points = data.map((item, index) => {
        const x = index * 50 + 25;
        const y = 100 - (item[yKey] / maxValue) * 80;

        // Create dots
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', 3);
        circle.setAttribute('fill', color);

        // Create tooltip for value
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${
          typeof item[xKey] === 'string' && item[xKey].includes('T')
            ? new Date(item[xKey]).toLocaleDateString()
            : item[xKey]
        }: ${item[yKey]}`;
        circle.appendChild(title);

        svg.appendChild(circle);

        // Add to path
        if (index === 0) {
          pathData = `M ${x} ${y}`;
        } else {
          pathData += ` L ${x} ${y}`;
        }

        return { x, y };
      });

      // Set path data
      path.setAttribute('d', pathData);
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', 2);
      path.setAttribute('fill', 'none');

      svg.appendChild(path);
      chartRef.current.appendChild(svg);

      // Create x-axis labels
      const labelsContainer = document.createElement('div');
      labelsContainer.className = styles.lineChartLabels;

      data.forEach((item, index) => {
        const label = document.createElement('span');
        label.className = styles.lineChartLabel;
        label.style.left = `${(index / (data.length - 1)) * 100}%`;
        label.textContent =
          typeof item[xKey] === 'string' && item[xKey].includes('T')
            ? new Date(item[xKey]).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
            : item[xKey];
        labelsContainer.appendChild(label);
      });

      chartRef.current.appendChild(labelsContainer);
    }
  }, [data, xKey, yKey, label, color]);

  return (
    <div className={styles.chartContainer} style={{ height: `${height}px` }}>
      <div ref={chartRef} className={styles.lineChartContainer}></div>
    </div>
  );
}
