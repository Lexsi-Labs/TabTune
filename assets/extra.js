/* Make the header title navigate home, matching the logo's behaviour. */
document$.subscribe(function () {
  var title = document.querySelector(".md-header__title");
  if (title && !title.dataset.ttHome) {
    title.dataset.ttHome = "1";
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      var logo = document.querySelector(".md-header__button.md-logo");
      if (logo && logo.href) {
        window.location.href = logo.href;
      }
    });
  }
});
