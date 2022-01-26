document.ondblclick = function (e) {
    // e.target, e.srcElement and e.toElement contains the element clicked.
    article = e.target.innerHTML

    const params = {
        "artical": article
    };
    const options = {
        method: 'POST',
        body: JSON.stringify(params),
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    };

    //Change this API url
    fetch('http://khoahockithuat.ap.ngrok.io/predict', options).then(response => response.json()).then(response => {
        if (parseFloat(response) > 0.5) {
            alert('Bài viết "' + article.substr(0, 10) + " ... " + article.substr(article.length - 10) + '" ' + (response * 100) + "% là tin thật")
        }
        else {
            alert('Bài viết "' + article.substr(0, 10) + " ... " + article.substr(article.length - 10) + '" ' + (100 - response * 100) + "% là tin giả")
        }
    });
};