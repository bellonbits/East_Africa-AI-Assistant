
    document.addEventListener('DOMContentLoaded', function() {
        const queryForm = document.getElementById('query-form');
        const queryInput = document.getElementById('query-input');
        const chatContainer = document.getElementById('chat-container');
        const resetBtn = document.getElementById('reset-btn');
        const imageGallery = document.getElementById('image-gallery');
        const imagesContainer = document.getElementById('images-container');
        const linksContainer = document.getElementById('links-container');
        
        // Function to add a message to the chat
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'mb-4 ' + (isUser ? 'text-right' : '');
            
            const bubble = document.createElement('div');
            bubble.className = isUser 
                ? 'bg-blue-100 p-3 rounded-lg inline-block max-w-3xl'
                : 'bg-green-100 p-3 rounded-lg inline-block max-w-3xl';
            
            if (isUser) {
                // For user messages, just add the text
                bubble.textContent = content;
            } else {
                // For assistant messages, render the markdown
                bubble.innerHTML = marked.parse(content);
                // Add target="_blank" to all links
                bubble.querySelectorAll('a').forEach(link => {
                    link.setAttribute('target', '_blank');
                });
            }
            
            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Function to display images
        function displayImages(images) {
            if (images && images.length > 0) {
                imageGallery.classList.remove('hidden');
                imagesContainer.innerHTML = '';
                
                images.forEach(image => {
                    const imgDiv = document.createElement('div');
                    imgDiv.className = 'aspect-w-16 aspect-h-9';
                    
                    const img = document.createElement('img');
                    img.src = image.thumbnail;
                    img.alt = image.title;
                    img.className = 'object-cover rounded w-full h-32';
                    
                    const link = document.createElement('a');
                    link.href = image.source_url;
                    link.target = '_blank';
                    link.appendChild(img);
                    
                    imgDiv.appendChild(link);
                    imagesContainer.appendChild(imgDiv);
                });
            } else {
                imageGallery.classList.add('hidden');
            }
        }
        
        // Function to display links
        function displayLinks(links) {
            if (links && links.length > 0) {
                linksContainer.innerHTML = '<ul class="list-disc pl-5 space-y-2"></ul>';
                const linksList = linksContainer.querySelector('ul');
                
                links.forEach(link => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = link.url;
                    a.className = 'text-blue-600 hover:underline';
                    a.target = '_blank';
                    a.textContent = link.title;
                    
                    li.appendChild(a);
                    
                    if (link.description) {
                        const desc = document.createElement('p');
                        desc.className = 'text-xs text-gray-600 mt-1';
                        desc.textContent = link.description;
                        li.appendChild(desc);
                    }
                    
                    linksList.appendChild(li);
                });
            }
        }
        
        // Handle form submission
        queryForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const query = queryInput.value.trim();
            if (!query) return;
            
            // Add user message to chat
            addMessage(query, true);
            
            // Clear input
            queryInput.value = '';
            
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'mb-4';
            loadingDiv.innerHTML = `
                <div class="bg-green-100 p-3 rounded-lg inline-block">
                    <p class="text-green-800">Thinking...</p>
                </div>
            `;
            chatContainer.appendChild(loadingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            try {
                // Send query to API
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to get response');
                }
                
                const data = await response.json();
                
                // Remove loading indicator
                chatContainer.removeChild(loadingDiv);
                
                // Add assistant response to chat
                addMessage(data.response);
                
                // Display images if available
                displayImages(data.images);
                
                // Display links if available
                displayLinks(data.links);
                
            } catch (error) {
                console.error('Error:', error);
                
                // Remove loading indicator
                chatContainer.removeChild(loadingDiv);
                
                // Show error message
                addMessage('Sorry, I encountered an error. Please try again.');
            }
        });
        
        // Handle reset button
        resetBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/reset', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    // Reload the page to start fresh
                    window.location.reload();
                }
            } catch (error) {
                console.error('Error resetting conversation:', error);
            }
        });
        
        // Load conversation history
        async function loadHistory() {
            try {
                // Get conversation ID from cookie
                const cookies = document.cookie.split(';').reduce((acc, cookie) => {
                    const [key, value] = cookie.trim().split('=');
                    acc[key] = value;
                    return acc;
                }, {});
                
                const conversationId = cookies['conversation_id'];
                
                if (!conversationId) return;
                
                const response = await fetch(`/history/${conversationId}`);
                if (!response.ok) return;
                
                const data = await response.json();
                
                if (data.history && data.history.length > 0) {
                    // Clear default welcome message
                    chatContainer.innerHTML = '';
                    
                    // Add messages from history
                    data.history.forEach(msg => {
                        addMessage(msg.content, msg.role === 'user');
                    });
                }
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        // Load history when page loads
        loadHistory();
    });
    