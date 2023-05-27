var map = L.map('map').setView([51.505, -0.09], 2);
var geoJsonData = null;

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

var geojsonLayer = L.geoJson().addTo(map);

// Fetch your GeoJSON data once
fetch('data.geojson')
    .then(response => response.json())
    .then(data => {
        geoJsonData = data;
        updateMap(document.getElementById("slider").value);
    });

function updateMap(year) {
    // Filter the GeoJSON data based on the year
    const filteredData = {
        ...geoJsonData,
        features: geoJsonData.features.filter(feature => feature.properties.year === parseInt(year))
    };
    geojsonLayer.clearLayers();
    geojsonLayer.addData(filteredData);

    // Display the current slider value
    document.getElementById("slider-value").innerText = year;
}
