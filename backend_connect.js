function predict(){ 
               
            let result1 = document.querySelector('.result1'); 
	    result1.innerHTML = 'Running Please Wait...'
            let buildingarea1 = document.querySelector('#BuildingArea1'); 
            let rooms1 = document.querySelector('#Rooms1'); 
            let postcode1 = document.querySelector('#Postcode1'); 
               
            // Creating a XHR object 
            let xhr = new XMLHttpRequest(); 
            let url = "http://localhost:5802/predict"; 
        
            // open a connection 
            xhr.open("POST", url, true); 
  
            // Set the request header i.e. which type of content you are sending 
            xhr.setRequestHeader("Content-Type", "application/json"); 
  
            // Create a state change callback 
            xhr.onreadystatechange = function () { 
                if (xhr.readyState === 4 && xhr.status === 200) { 
  
                    // Print received data from server  
                    result1.innerHTML = this.responseText;
  
                } 
            }; 
  
            // Converting JSON data to string 
            var data = JSON.stringify({ "BuildingArea": buildingarea1.value, "Rooms": rooms1.value, "Postcode": postcode1.value}); 
  
            // Sending data with the request 
            xhr.send(data); 
        } 

function retrain(){ 
               
            let result2 = document.querySelector('.result2');
            result2.innerHTML = 'Running Please Wait...'
            let buildingarea2 = document.querySelector('#BuildingArea2'); 
            let rooms2 = document.querySelector('#Rooms2'); 
            let postcode2 = document.querySelector('#Postcode2'); 
            let price2 = document.querySelector('#Price2');
               
            // Creating a XHR object 
            let xhr = new XMLHttpRequest(); 
            let url = "http://localhost:5802/retrain"; 
        
            // open a connection 
            xhr.open("POST", url, true); 
  
            // Set the request header i.e. which type of content you are sending 
            xhr.setRequestHeader("Content-Type", "application/json"); 
  
            // Create a state change callback 
            xhr.onreadystatechange = function () { 
                if (xhr.readyState === 4 && xhr.status === 200) { 
  
                    // Print received data from server  
                    result2.innerHTML = this.responseText;
  
                } 
            }; 
  
            // Converting JSON data to string 
            var data = JSON.stringify({ "BuildingArea": buildingarea2.value, "Rooms": rooms2.value, "Postcode": postcode2.value, "Price": price2.value}); 
  
            // Sending data with the request 
            xhr.send(data); 
        } 

function reset(){ 
               
            let result3 = document.querySelector('.result3');
            result3.innerHTML = 'Running Please Wait...' 
            let dummy = " ";
               
            // Creating a XHR object 
            let xhr = new XMLHttpRequest(); 
            let url = "http://localhost:5802/reset"; 
        
            // open a connection 
            xhr.open("POST", url, true); 
  
            // Set the request header i.e. which type of content you are sending 
            xhr.setRequestHeader("Content-Type", "application/json"); 
  
            // Create a state change callback 
            xhr.onreadystatechange = function () { 
                if (xhr.readyState === 4 && xhr.status === 200) { 
  
                    // Print received data from server  
                    result3.innerHTML = this.responseText;
  
                } 
            }; 
  
            // Converting JSON data to string 
            var data = JSON.stringify({ "dummy": dummy.value}); 
  
            // Sending data with the request 
            xhr.send(data); 
        } 
