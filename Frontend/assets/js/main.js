(function() {
  "use strict";

  /**
   * Apply .scrolled class to the body as the page is scrolled down
   */
  function toggleScrolled() {
    const selectBody = document.querySelector('body');
    const selectHeader = document.querySelector('#header');
    if (!selectHeader.classList.contains('scroll-up-sticky') && !selectHeader.classList.contains('sticky-top') && !selectHeader.classList.contains('fixed-top')) return;
    window.scrollY > 100 ? selectBody.classList.add('scrolled') : selectBody.classList.remove('scrolled');
  }

  document.addEventListener('scroll', toggleScrolled);
  window.addEventListener('load', toggleScrolled);

  /**
   * Mobile nav toggle
   */
  const mobileNavToggleBtn = document.querySelector('.mobile-nav-toggle');

  function mobileNavToogle() {
    document.querySelector('body').classList.toggle('mobile-nav-active');
    mobileNavToggleBtn.classList.toggle('bi-list');
    mobileNavToggleBtn.classList.toggle('bi-x');
  }
  if (mobileNavToggleBtn) {
    mobileNavToggleBtn.addEventListener('click', mobileNavToogle);
  }

  /**
   * Hide mobile nav on same-page/hash links
   */
  document.querySelectorAll('#navmenu a').forEach(navmenu => {
    navmenu.addEventListener('click', () => {
      if (document.querySelector('.mobile-nav-active')) {
        mobileNavToogle();
      }
    });

  });

  /**
   * Toggle mobile nav dropdowns
   */
  document.querySelectorAll('.navmenu .toggle-dropdown').forEach(navmenu => {
    navmenu.addEventListener('click', function(e) {
      e.preventDefault();
      this.parentNode.classList.toggle('active');
      this.parentNode.nextElementSibling.classList.toggle('dropdown-active');
      e.stopImmediatePropagation();
    });
  });

  /**
   * Scroll top button
   */
  let scrollTop = document.querySelector('.scroll-top');

  function toggleScrollTop() {
    if (scrollTop) {
      window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }
  }
  scrollTop.addEventListener('click', (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  window.addEventListener('load', toggleScrollTop);
  document.addEventListener('scroll', toggleScrollTop);

  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  window.addEventListener('load', aosInit);

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Frequently Asked Questions Toggle
   */
  document.querySelectorAll('.faq-item h3, .faq-item .faq-toggle').forEach((faqItem) => {
    faqItem.addEventListener('click', () => {
      faqItem.parentNode.classList.toggle('faq-active');
    });
  });

  /**
   * Init isotope layout and filters
   */
  document.querySelectorAll('.isotope-layout').forEach(function(isotopeItem) {
    let layout = isotopeItem.getAttribute('data-layout') ?? 'masonry';
    let filter = isotopeItem.getAttribute('data-default-filter') ?? '*';
    let sort = isotopeItem.getAttribute('data-sort') ?? 'original-order';

    let initIsotope;
    imagesLoaded(isotopeItem.querySelector('.isotope-container'), function() {
      initIsotope = new Isotope(isotopeItem.querySelector('.isotope-container'), {
        itemSelector: '.isotope-item',
        layoutMode: layout,
        filter: filter,
        sortBy: sort
      });
    });

    isotopeItem.querySelectorAll('.isotope-filters li').forEach(function(filters) {
      filters.addEventListener('click', function() {
        isotopeItem.querySelector('.isotope-filters .filter-active').classList.remove('filter-active');
        this.classList.add('filter-active');
        initIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        if (typeof aosInit === 'function') {
          aosInit();
        }
      }, false);
    });

  });

  /**
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function(swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener('load', function(e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: 'smooth'
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu Scrollspy
   */
  let navmenulinks = document.querySelectorAll('.navmenu a');

  function navmenuScrollspy() {
    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
        navmenulink.classList.add('active');
      } else {
        navmenulink.classList.remove('active');
      }
    })
  }
  window.addEventListener('load', navmenuScrollspy);
  document.addEventListener('scroll', navmenuScrollspy);

  /**
   * Face recognition upload and prediction
   */
  function initFaceRecognition() {
    /** @type {HTMLElement|null} */
    const uploadArea = document.getElementById('upload-area');
    /** @type {HTMLInputElement|null} */
    const imageInput = /** @type {HTMLInputElement|null} */ (document.getElementById('image-input'));
    /** @type {HTMLElement|null} */
    const previewContainer = document.getElementById('preview-container');
    /** @type {HTMLImageElement|null} */
    const previewImage = /** @type {HTMLImageElement|null} */ (document.getElementById('preview-image'));
    /** @type {HTMLButtonElement|null} */
    const clearBtn = /** @type {HTMLButtonElement|null} */ (document.getElementById('clear-btn'));
    /** @type {HTMLElement|null} */
    const loadingSpinner = document.getElementById('loading-spinner');
    /** @type {HTMLElement|null} */
    const resultsContainer = document.getElementById('results-container');
    /** @type {HTMLElement|null} */
    const errorContainer = document.getElementById('error-container');
    /** @type {HTMLElement|null} */
    const peopleList = document.getElementById('people-list');

    if (!uploadArea || !imageInput || !previewContainer || !previewImage) {
      return;
    }

    /** @param {HTMLElement|null} element */
    const showElement = (element) => {
      if (element) {
        element.classList.remove('d-none');
      }
    };

    /** @param {HTMLElement|null} element */
    const hideElement = (element) => {
      if (element) {
        element.classList.add('d-none');
      }
    };

    const resetMessages = () => {
      hideElement(loadingSpinner);
      hideElement(resultsContainer);
      hideElement(errorContainer);
    };

    /** @param {string} message */
    const showError = (message) => {
      /** @type {HTMLElement|null} */
      const errorMessage = document.getElementById('error-message');
      if (errorMessage) {
        errorMessage.textContent = message;
      }
      hideElement(loadingSpinner);
      hideElement(resultsContainer);
      showElement(errorContainer);
    };

    /** @param {string[]} labels */
    const renderPeople = (labels) => {
      if (!peopleList) return;
      peopleList.innerHTML = labels.map((label) => `
        <div class="col-6 col-md-4">
          <div class="people-item">
            <div class="person-avatar">${label.charAt(0).toUpperCase()}</div>
            <div class="person-name">${label}</div>
          </div>
        </div>
      `).join('');
    };

    const loadPeopleDatabase = async () => {
      if (!peopleList) return;
      try {
        const response = await fetch('/api/labels');
        const data = await response.json();
        if (data.success && Array.isArray(data.labels)) {
          renderPeople(data.labels);
          return;
        }
        peopleList.innerHTML = '<div class="col-12 text-muted">No labels were returned by the backend.</div>';
      } catch (error) {
        peopleList.innerHTML = '<div class="col-12 text-muted">Connect the Flask backend to load the database.</div>';
      }
    };

    /** @param {File} file */
    const predictImage = async (file) => {
      const formData = new FormData();
      formData.append('image', file);
      resetMessages();
      showElement(loadingSpinner);

      try {
        const response = await fetch('/api/recognize', {
          method: 'POST',
          body: formData,
        });
        /** @type {{ success?: boolean, error?: string, person?: string, confidence?: number }} */
        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || 'Recognition failed');
        }

        /** @type {HTMLElement|null} */
        const personName = document.getElementById('person-name');
        /** @type {HTMLElement|null} */
        const confidenceLevel = document.getElementById('confidence-level');
        /** @type {HTMLElement|null} */
        const confidenceBar = document.getElementById('confidence-bar');

        if (personName) {
          personName.textContent = data.person || 'Unknown';
        }
        if (confidenceLevel) {
          confidenceLevel.textContent = `${Math.round(data.confidence || 0)}%`;
        }
        if (confidenceBar) {
          confidenceBar.style.width = `${Math.round(data.confidence || 0)}%`;
        }

        hideElement(loadingSpinner);
        showElement(resultsContainer);
      } catch (error) {
        showError(error instanceof Error ? error.message : 'Unable to predict the image');
      }
    };

    const clearImage = () => {
      imageInput.value = '';
      uploadArea.classList.remove('d-none');
      hideElement(previewContainer);
      resetMessages();
    };

    uploadArea.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (event) => {
      const target = /** @type {HTMLInputElement} */ (event.currentTarget);
      const [file] = target.files || [];
      if (!file) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (readerEvent) => {
        if (readerEvent.target instanceof FileReader && readerEvent.target.result && previewImage) {
          previewImage.src = String(readerEvent.target.result);
        }
        uploadArea.classList.add('d-none');
        showElement(previewContainer);
        predictImage(file);
      };
      reader.readAsDataURL(file);
    });

    uploadArea.addEventListener('dragover', (event) => {
      event.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (event) => {
      event.preventDefault();
      uploadArea.classList.remove('dragover');
      const [file] = event.dataTransfer ? event.dataTransfer.files : [];
      if (!file || !file.type.startsWith('image/')) {
        showError('Please drop a valid image file.');
        return;
      }

      const reader = new FileReader();
      reader.onload = (readerEvent) => {
        if (readerEvent.target instanceof FileReader && readerEvent.target.result && previewImage) {
          previewImage.src = String(readerEvent.target.result);
        }
        uploadArea.classList.add('d-none');
        showElement(previewContainer);
        predictImage(file);
      };
      reader.readAsDataURL(file);
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', clearImage);
    }

    loadPeopleDatabase();
  }

  /**
   * Load and display model evaluation metrics
   */
  function loadModelMetrics() {
    const metricsLoading = document.getElementById('metrics-loading');
    const metricsContent = document.getElementById('metrics-content');
    const metricsError = document.getElementById('metrics-error');

    if (!metricsLoading || !metricsContent) {
      return;
    }

    fetch('/api/model-metrics')
      .then(response => response.json())
      .then(data => {
        if (!data.success || !data.available) {
          throw new Error(data.error || 'Metrics not available');
        }

        // Display accuracy metrics
        document.getElementById('train-accuracy').textContent = 
          (data.train_accuracy * 100).toFixed(2) + '%';
        document.getElementById('test-accuracy').textContent = 
          (data.test_accuracy * 100).toFixed(2) + '%';
        
        // Display model info
        const modelInfoEl = document.getElementById('model-info');
        if (modelInfoEl && data.model_info) {
          modelInfoEl.innerHTML = `
            <strong>PCA Components:</strong> ${data.model_info.n_components}<br>
            <strong>KNN Neighbors:</strong> ${data.model_info.n_neighbors}
          `;
        }

        // Display per-class metrics
        const tableBody = document.getElementById('class-metrics-table');
        if (tableBody && data.class_metrics) {
          tableBody.innerHTML = '';
          
          Object.keys(data.class_metrics).forEach(classId => {
            const metric = data.class_metrics[classId];
            const row = document.createElement('tr');
            row.innerHTML = `
              <td><strong>${metric.label}</strong></td>
              <td>${metric.samples}</td>
              <td>
                <span class="badge" style="background-color: ${metric.accuracy >= 0.9 ? '#28a745' : metric.accuracy >= 0.7 ? '#ffc107' : '#dc3545'}">
                  ${(metric.accuracy * 100).toFixed(1)}%
                </span>
              </td>
            `;
            tableBody.appendChild(row);
          });
        }

        // Show content and hide loading
        metricsLoading.style.display = 'none';
        metricsContent.style.display = 'block';
        metricsError.style.display = 'none';

      })
      .catch(error => {
        console.error('Error loading metrics:', error);
        metricsLoading.style.display = 'none';
        metricsContent.style.display = 'none';
        metricsError.style.display = 'block';
        
        const errorText = document.getElementById('metrics-error-text');
        if (errorText) {
          errorText.textContent = error.message || 'Could not load model metrics';
        }
      });
  }

  window.addEventListener('load', initFaceRecognition);
  window.addEventListener('load', loadModelMetrics);

})();