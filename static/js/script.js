document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.querySelector('.file-input input[type="file"]');
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        fileInput.closest("form").submit();
      }
    });
  }
});
