document.getElementById('health-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const formData = new FormData(event.target);
    
    fetch('http://localhost:5000/BrainStrokePredict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        
        // Hide the form
        document.getElementById('health-form').style.display = 'none';

        // Create a new element to display the prediction result
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result';
        resultDiv.innerHTML = `
            <h2>Prediction Result</h2>
            <p>${data.prediction}</p>
        `;
        
        // Append the result to the container
        document.querySelector('.container').appendChild(resultDiv);

        // Check if prediction is 1 and show the upload button
        if (data.prediction == "Person is at risk of Brain Stroke") {
            const uploadButton = document.createElement('a');
            uploadButton.href = "/BrainStrokeImageForm"; // Update to your actual upload page
            uploadButton.innerHTML = '<button class="button">Upload CT Scan</button>';
            resultDiv.appendChild(uploadButton);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while submitting the form.');
    });
});
