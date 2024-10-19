document.getElementById('health-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const formData = new FormData(event.target);
    
    fetch('http://localhost:5000/submit', {
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
            <p>${JSON.stringify(data.prediction)}</p>
            <p>${data.message}</p>
        `;
        
        // Append the result to the container
        document.querySelector('.container').appendChild(resultDiv);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while submitting the form.');
    });
});
