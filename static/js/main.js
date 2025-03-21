
// // #_________________________working live stream_____________________________________#
// document.addEventListener("DOMContentLoaded", function () {
//   const uploadImages = document.getElementById("uploadImages");
//   const uploadVideos = document.getElementById("uploadVideos");
//   const processButton = document.getElementById("processButton");
//   const previewArea = document.getElementById("previewArea");
//   const detectionOutput = document.getElementById("detectionOutput");
//   const trackingOutput = document.getElementById("trackingOutput");
//   const enhancementOutput = document.getElementById("enhancementOutput");
//   const startStreamButton = document.getElementById("startStream");
//   const stopStreamButton = document.getElementById("stopStream");
//   const ipAddressInput = document.getElementById("ipAddress");
//   const refreshIPButton = document.getElementById("refreshIP");

//   let selectedFiles = new FormData();
//   let streamActive = false;
//   let streamImages = {
//     preview: null,
//     detection: null,
//     tracking: null,
//     enhancement: null,
//   };

//   // Function to create a file input
//   function createFileInput(accept) {
//     const input = document.createElement("input");
//     input.type = "file";
//     input.multiple = true;
//     input.accept = accept;
//     input.style.display = "none";
//     return input;
//   }

//   // Function to create and style stream image
//   function createStreamImage() {
//     const img = document.createElement("img");
//     img.style.maxWidth = "100%";
//     img.style.height = "auto";
//     img.style.marginBottom = "10px";
//     img.style.borderRadius = "4px";
//     img.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
//     return img;
//   }

//   // Handle IP refresh
//   refreshIPButton.addEventListener("click", () => {
//     ipAddressInput.value = "";
//   });

//   // Handle start streaming
//   startStreamButton.addEventListener("click", async () => {
//     const ipAddress = ipAddressInput.value.trim();
//     if (!ipAddress) {
//       alert("Please enter a valid IP address");
//       return;
//     }

//     try {
//       startStreamButton.disabled = true;
//       stopStreamButton.disabled = false;

//       const response = await fetch(`/api/start_stream?ip=${ipAddress}`);
//       if (!response.ok) {
//         throw new Error(`Failed to start stream: ${response.statusText}`);
//       }

//       const data = await response.json();
//       streamActive = true;

//       // Clear previous content
//       previewArea.innerHTML = "";
//       detectionOutput.innerHTML = "";
//       trackingOutput.innerHTML = "";
//       enhancementOutput.innerHTML = "";

//       // Create and set up stream images
//       streamImages.preview = createStreamImage();
//       streamImages.detection = createStreamImage();
//       streamImages.tracking = createStreamImage();
//       streamImages.enhancement = createStreamImage();

//       // Add images to their containers
//       previewArea.appendChild(streamImages.preview);
//       detectionOutput.appendChild(streamImages.detection);
//       trackingOutput.appendChild(streamImages.tracking);
//       enhancementOutput.appendChild(streamImages.enhancement);

//       // Start streams
//       Object.entries(data.streams).forEach(([type, url]) => {
//         const img = streamImages[type];
//         if (img) {
//           const updateImage = () => {
//             if (streamActive) {
//               img.src = `${url}?t=${Date.now()}`;
//               setTimeout(updateImage, 50); // Update every 50ms
//             }
//           };
//           updateImage();
//         }
//       });
//     } catch (error) {
//       console.error("Streaming error:", error);
//       alert(
//         "Failed to start streaming. Please check the IP address and try again."
//       );
//       startStreamButton.disabled = false;
//       stopStreamButton.disabled = true;
//     }
//   });

//   // Handle stop streaming
//   stopStreamButton.addEventListener("click", async () => {
//     try {
//       streamActive = false;
//       stopStreamButton.disabled = true;
//       startStreamButton.disabled = false;

//       const response = await fetch("/api/stop_stream");
//       if (!response.ok) {
//         throw new Error(`Failed to stop stream: ${response.statusText}`);
//       }

//       // Clear stream displays
//       previewArea.innerHTML = "";
//       detectionOutput.innerHTML = "<p>No results available</p>";
//       trackingOutput.innerHTML = "<p>No tracking results yet</p>";
//       enhancementOutput.innerHTML = "<p>No enhancement results yet</p>";

//       // Reset stream images
//       streamImages = {
//         preview: null,
//         detection: null,
//         tracking: null,
//         enhancement: null,
//       };
//     } catch (error) {
//       console.error("Error stopping stream:", error);
//       alert("Error stopping stream. Please try again.");
//     }
//   });

//   // Handle image upload
//   uploadImages.addEventListener("click", () => {
//     const input = createFileInput("image/*");
//     input.addEventListener("change", (e) => handleFileSelect(e, "images[]"));
//     input.click();
//   });

//   // Handle video upload
//   uploadVideos.addEventListener("click", () => {
//     const input = createFileInput("video/*");
//     input.addEventListener("change", (e) => handleFileSelect(e, "videos[]"));
//     input.click();
//   });

//   // Handle file selection
//   function handleFileSelect(event, inputName) {
//     const files = event.target.files;
//     previewArea.innerHTML = "";

//     // Clear previous files of the same type
//     selectedFiles = new FormData();

//     Array.from(files).forEach((file) => {
//       selectedFiles.append(inputName, file);

//       if (file.type.startsWith("image/")) {
//         const img = document.createElement("img");
//         img.src = URL.createObjectURL(file);
//         img.style.maxWidth = "200px";
//         img.style.margin = "10px";
//         img.loading = "lazy";
//         previewArea.appendChild(img);
//       } else if (file.type.startsWith("video/")) {
//         const video = document.createElement("video");
//         video.src = URL.createObjectURL(file);
//         video.controls = true;
//         video.style.maxWidth = "200px";
//         video.style.margin = "10px";
//         previewArea.appendChild(video);
//       }
//     });
//   }

//   // Handle process button click
//   processButton.addEventListener("click", async () => {
//     if (selectedFiles.entries().next().done) {
//       alert("Please select files first");
//       return;
//     }

//     try {
//       processButton.disabled = true;
//       processButton.textContent = "Processing...";

//       // Clear previous outputs before processing
//       detectionOutput.innerHTML = "<p>Processing...</p>";
//       trackingOutput.innerHTML = "<p>Processing...</p>";
//       enhancementOutput.innerHTML = "<p>Processing...</p>";

//       const response = await fetch("/api/process", {
//         method: "POST",
//         body: selectedFiles,
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();

//       // Clear the processing messages
//       detectionOutput.innerHTML = "";
//       trackingOutput.innerHTML = "";
//       enhancementOutput.innerHTML = "";

//       updateOutputs(data);
//     } catch (error) {
//       console.error("Error processing files:", error);
//       detectionOutput.innerHTML = "<p>Error processing files</p>";
//       trackingOutput.innerHTML = "<p>Error processing files</p>";
//       enhancementOutput.innerHTML = "<p>Error processing files</p>";
//       alert("Error processing files. Please try again.");
//     } finally {
//       processButton.disabled = false;
//       processButton.textContent = "Process";
//     }
//   });

//   // Function to create image element
//   function createImageElement(imageUrl) {
//     const img = document.createElement("img");
//     img.src = imageUrl;
//     img.style.maxWidth = "100%";
//     img.style.height = "auto";
//     img.style.marginBottom = "10px";
//     img.style.borderRadius = "4px";
//     img.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
//     return img;
//   }

//   // Function to create video element
//   function createVideoElement(videoUrl) {
//     const video = document.createElement("video");
//     video.src = videoUrl;
//     video.controls = true;
//     video.style.maxWidth = "100%";
//     video.style.height = "auto";
//     video.style.marginBottom = "10px";
//     video.style.borderRadius = "4px";
//     video.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
//     return video;
//   }

//   // Function to update outputs
//   function updateOutputs(data) {
//     // Clear previous outputs
//     detectionOutput.innerHTML = "";
//     trackingOutput.innerHTML = "";
//     enhancementOutput.innerHTML = "";

//     // Handle processed images
//     if (data.processedImages && data.processedImages.length > 0) {
//       data.processedImages.forEach((imageData) => {
//         if (imageData.detection) {
//           detectionOutput.appendChild(createImageElement(imageData.detection));
//         }
//         if (imageData.tracking) {
//           trackingOutput.appendChild(createImageElement(imageData.tracking));
//         }
//         if (imageData.enhancement) {
//           enhancementOutput.appendChild(
//             createImageElement(imageData.enhancement)
//           );
//         }
//       });
//     }

//     // Handle processed videos
//     if (data.processedVideos && data.processedVideos.length > 0) {
//       data.processedVideos.forEach((videoData) => {
//         if (videoData.detection) {
//           detectionOutput.appendChild(createVideoElement(videoData.detection));
//         }
//         if (videoData.tracking) {
//           trackingOutput.appendChild(createVideoElement(videoData.tracking));
//         }
//         if (videoData.enhancement) {
//           enhancementOutput.appendChild(
//             createVideoElement(videoData.enhancement)
//           );
//         }
//       });
//     }

//     // Add "No results" message if no content
//     [detectionOutput, trackingOutput, enhancementOutput].forEach((output) => {
//       if (!output.hasChildNodes()) {
//         output.innerHTML = "<p>No results available</p>";
//       }
//     });
//   }
// });













// ___________________________working fine and also displaying the output on the UI using DIAT camera__________________________



document.addEventListener("DOMContentLoaded", function () {
  const uploadImages = document.getElementById("uploadImages");
  const uploadVideos = document.getElementById("uploadVideos");
  const processButton = document.getElementById("processButton");
  const previewArea = document.getElementById("previewArea");
  const detectionOutput = document.getElementById("detectionOutput");
  const trackingOutput = document.getElementById("trackingOutput");
  const enhancementOutput = document.getElementById("enhancementOutput");
  const startStreamButton = document.getElementById("startStream");
  const stopStreamButton = document.getElementById("stopStream");
  const ipAddressInput = document.getElementById("ipAddress");
  const refreshIPButton = document.getElementById("refreshIP");

  let selectedFiles = new FormData();
  let streamActive = false;
  let streamImages = {
    preview: null,
    detection: null,
    tracking: null,
    enhancement: null,
  };

  // Function to create a file input
  function createFileInput(accept) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = accept;
    input.style.display = "none";
    return input;
  }

  // Function to create and style stream image
  function createStreamImage() {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.height = "auto";
    img.style.marginBottom = "10px";
    img.style.borderRadius = "4px";
    img.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
    return img;
  }

  // Handle IP refresh
  refreshIPButton.addEventListener("click", () => {
    ipAddressInput.value = "";
  });

  // Handle start streaming
  startStreamButton.addEventListener("click", async () => {
    const ipAddress = ipAddressInput.value.trim();
    if (!ipAddress) {
      alert("Please enter a valid IP address");
      return;
    }

    try {
      startStreamButton.disabled = true;
      stopStreamButton.disabled = false;

      const response = await fetch(`/api/start_stream?ip=${ipAddress}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to start stream: ${response.statusText}`);
      }

      const data = await response.json();
      streamActive = true;

      // Clear previous content
      previewArea.innerHTML = "";
      detectionOutput.innerHTML = "";
      trackingOutput.innerHTML = "";
      enhancementOutput.innerHTML = "";

      // Create and set up stream images
      streamImages.preview = createStreamImage();
      streamImages.detection = createStreamImage();
      streamImages.tracking = createStreamImage();
      streamImages.enhancement = createStreamImage();

      // Add images to their containers
      previewArea.appendChild(streamImages.preview);
      detectionOutput.appendChild(streamImages.detection);
      trackingOutput.appendChild(streamImages.tracking);
      enhancementOutput.appendChild(streamImages.enhancement);

      // Start streams - use direct URLs instead of updating with timestamps
      Object.entries(data.streams).forEach(([type, url]) => {
        const img = streamImages[type];
        if (img) {
          // Set the source directly to the MJPEG stream
          img.src = url;
          
          // Add error handling for stream images
          img.onerror = function() {
            console.error(`Error loading ${type} stream`);
            // Try to reload after a short delay
            setTimeout(() => {
              if (streamActive) {
                img.src = url + "?t=" + Date.now();
              }
            }, 1000);
          };
        }
      });
    } catch (error) {
      console.error("Streaming error:", error);
      alert(
        error.message || "Failed to start streaming. Please check the IP address and try again."
      );
      startStreamButton.disabled = false;
      stopStreamButton.disabled = true;
    }
  });

  // Handle stop streaming
  stopStreamButton.addEventListener("click", async () => {
    try {
      streamActive = false;
      stopStreamButton.disabled = true;
      startStreamButton.disabled = false;

      const response = await fetch("/api/stop_stream");
      if (!response.ok) {
        throw new Error(`Failed to stop stream: ${response.statusText}`);
      }

      // Clear stream displays
      previewArea.innerHTML = "";
      detectionOutput.innerHTML = "<p>No results available</p>";
      trackingOutput.innerHTML = "<p>No tracking results yet</p>";
      enhancementOutput.innerHTML = "<p>No enhancement results yet</p>";

      // Reset stream images
      streamImages = {
        preview: null,
        detection: null,
        tracking: null,
        enhancement: null,
      };
    } catch (error) {
      console.error("Error stopping stream:", error);
      alert("Error stopping stream. Please try again.");
    }
  });

  // Handle image upload
  uploadImages.addEventListener("click", () => {
    const input = createFileInput("image/*");
    input.addEventListener("change", (e) => handleFileSelect(e, "images[]"));
    input.click();
  });

  // Handle video upload
  uploadVideos.addEventListener("click", () => {
    const input = createFileInput("video/*");
    input.addEventListener("change", (e) => handleFileSelect(e, "videos[]"));
    input.click();
  });

  // Handle file selection
  function handleFileSelect(event, inputName) {
    const files = event.target.files;
    previewArea.innerHTML = "";

    // Clear previous files of the same type
    selectedFiles = new FormData();

    Array.from(files).forEach((file) => {
      selectedFiles.append(inputName, file);

      if (file.type.startsWith("image/")) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.style.maxWidth = "200px";
        img.style.margin = "10px";
        img.loading = "lazy";
        previewArea.appendChild(img);
      } else if (file.type.startsWith("video/")) {
        const video = document.createElement("video");
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.style.maxWidth = "200px";
        video.style.margin = "10px";
        previewArea.appendChild(video);
      }
    });
  }

  // Handle process button click
  processButton.addEventListener("click", async () => {
    if (selectedFiles.entries().next().done) {
      alert("Please select files first");
      return;
    }

    try {
      processButton.disabled = true;
      processButton.textContent = "Processing...";

      // Clear previous outputs before processing
      detectionOutput.innerHTML = "<p>Processing...</p>";
      trackingOutput.innerHTML = "<p>Processing...</p>";
      enhancementOutput.innerHTML = "<p>Processing...</p>";

      const response = await fetch("/api/process", {
        method: "POST",
        body: selectedFiles,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Clear the processing messages
      detectionOutput.innerHTML = "";
      trackingOutput.innerHTML = "";
      enhancementOutput.innerHTML = "";

      updateOutputs(data);
    } catch (error) {
      console.error("Error processing files:", error);
      detectionOutput.innerHTML = "<p>Error processing files</p>";
      trackingOutput.innerHTML = "<p>Error processing files</p>";
      enhancementOutput.innerHTML = "<p>Error processing files</p>";
      alert("Error processing files. Please try again.");
    } finally {
      processButton.disabled = false;
      processButton.textContent = "Process";
    }
  });

  // Function to create image element
  function createImageElement(imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.style.maxWidth = "100%";
    img.style.height = "auto";
    img.style.marginBottom = "10px";
    img.style.borderRadius = "4px";
    img.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
    return img;
  }

  // Function to create video element
  function createVideoElement(videoUrl) {
    const video = document.createElement("video");
    video.src = videoUrl;
    video.controls = true;
    video.style.maxWidth = "100%";
    video.style.height = "auto";
    video.style.marginBottom = "10px";
    video.style.borderRadius = "4px";
    video.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.1)";
    return video;
  }

  // Function to update outputs
  function updateOutputs(data) {
    // Clear previous outputs
    detectionOutput.innerHTML = "";
    trackingOutput.innerHTML = "";
    enhancementOutput.innerHTML = "";

    // Handle processed images
    if (data.processedImages && data.processedImages.length > 0) {
      data.processedImages.forEach((imageData) => {
        if (imageData.detection) {
          detectionOutput.appendChild(createImageElement(imageData.detection));
        }
        if (imageData.tracking) {
          trackingOutput.appendChild(createImageElement(imageData.tracking));
        }
        if (imageData.enhancement) {
          enhancementOutput.appendChild(
            createImageElement(imageData.enhancement)
          );
        }
      });
    }

    // Handle processed videos
    if (data.processedVideos && data.processedVideos.length > 0) {
      data.processedVideos.forEach((videoData) => {
        if (videoData.detection) {
          detectionOutput.appendChild(createVideoElement(videoData.detection));
        }
        if (videoData.tracking) {
          trackingOutput.appendChild(createVideoElement(videoData.tracking));
        }
        if (videoData.enhancement) {
          enhancementOutput.appendChild(
            createVideoElement(videoData.enhancement)
          );
        }
      });
    }

    // Add "No results" message if no content
    [detectionOutput, trackingOutput, enhancementOutput].forEach((output) => {
      if (!output.hasChildNodes()) {
        output.innerHTML = "<p>No results available</p>";
      }
    });
  }
});


