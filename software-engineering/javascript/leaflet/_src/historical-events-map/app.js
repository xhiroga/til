var map = L.map('map').setView([51.505, -0.09], 2);
var geoJsonData = null;

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

fetch('data.geojson')
    .then(response => response.json())
    .then(data => {
        geoJsonData = data;
        updateMap(document.getElementById("slider").value);
    });

var geojsonLayer = L.geoJson(null, {
    pointToLayer: function (feature, latlng) {
        if (feature.properties.image) {
            var icon = L.icon({
                iconUrl: feature.properties.image,
                iconSize: [38, 38], // size of the icon
                iconAnchor: [22, 94], // point of the icon which will correspond to marker's location
                popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
            });
            return L.marker(latlng, {icon: icon});
        } else {
            return L.marker(latlng);
        }
    }
}).addTo(map);

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
