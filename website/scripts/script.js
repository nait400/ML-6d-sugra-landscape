const navToggle = document.querySelector('.nav-toggle');
const sidebar = document.querySelector('.sidebar');

navToggle.addEventListener('click', () => {
    sidebar.classList.toggle('hidden');
});


const navLinks = document.querySelectorAll('.nav-link');
const contentHeaders = document.querySelectorAll('.content-header');

navLinks.forEach(link => {
    link.addEventListener('click', () => {
        const content = link.nextElementSibling;
        content.style.display = content.style.display === 'block' ? 'none' : 'block';
    });
});

contentHeaders.forEach(header => {
    header.addEventListener('click', () => {
        const content = header.nextElementSibling;
        content.style.display = content.style.display === 'block' ? 'none' : 'block';
    });
});


const compareBtn = document.getElementById('compareBtn');
const iframeContainer = document.getElementById('iframeContainer');
const incrementBtns = document.querySelectorAll('.incrementBtn');

incrementBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetInput = document.getElementById(btn.dataset.target);
        let currentValue = parseInt(targetInput.value);
        if (currentValue < 10) {
            targetInput.value = currentValue + 1;
        }
    });
});

compareBtn.addEventListener('click', () => {
    iframeContainer.innerHTML = ''; // Clear previous iframes

    const pageInputs = document.querySelectorAll('.page-list input[type="number"]');
    pageInputs.forEach(input => {
        const pageCount = parseInt(input.value);
        const pageUrl = 'ClusterAnalysis_GramAE_LC-02-42/' + input.id + '.html'; // Assuming page URLs are based on IDs
        for (let i = 0; i < pageCount; i++) {
            const iframe = document.createElement('iframe');
            iframe.src = pageUrl;
            iframeContainer.appendChild(iframe);
        }
    });
});

const resetBtn = document.getElementById('resetBtn');

resetBtn.addEventListener('click', () => {
    const pageInputs = document.querySelectorAll('.page-selection input[type="number"]');
    pageInputs.forEach(input => {
        input.value = 0; // Reset input values to 0
    });
    iframeContainer.innerHTML = ''; // Clear the iframe container
});