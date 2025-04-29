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

function displayComparisons() {
    iframeContainer.innerHTML = ''; // Clear previous iframes

    const selectedPages = document.querySelectorAll('.page-list input:checked');
    selectedPages.forEach(page => {
        const iframe = document.createElement('iframe');
        iframe.src = page.value;
        iframeContainer.appendChild(iframe);
    });
}

compareBtn.addEventListener('click', displayComparisons);