let sections = document.querySelectorAll('section');
let navLinks = document.querySelectorAll('.navbar-nav a');

window.onscroll = () => {
    let scrollPosition = document.documentElement.scrollTop || document.body.scrollTop;

    sections.forEach(sec => {
        let offset = sec.offsetTop - document.querySelector('.navbar').offsetHeight;
        let height = sec.offsetHeight;
        let id = sec.getAttribute('id');

        if (scrollPosition >= offset && scrollPosition < offset + height) {
            navLinks.forEach(link => {
                link.classList.remove('active');
            });
            document.querySelector('.navbar-nav a[href="#' + id + '"]').classList.add('active');
        }
    });
};