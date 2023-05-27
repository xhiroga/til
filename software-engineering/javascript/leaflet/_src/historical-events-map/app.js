const map = L.map('map').setView([51.505, -0.09], 2);
let geoJsonData = null;

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

var geojsonLayer = L.geoJson().addTo(map);

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

    // Clear all layers
    map.eachLayer(function (layer) {
      if (layer !== L.tileLayer) map.removeLayer(layer);
    });

    // Add the filtered data to the map
    filteredData.features.forEach(function(feature) {
      var coordinates = feature.geometry.coordinates;
      var image = feature.properties.image;
      if (image) {
        var icon = L.icon({
          iconUrl: image,
          iconSize: [38, 38], // size of the icon
          iconAnchor: [22, 94], // point of the icon which will correspond to marker's location
          popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
        });
        L.marker([coordinates[1], coordinates[0]], {icon: icon}).addTo(map);
      } else {
        L.marker([coordinates[1], coordinates[0]]).addTo(map);
      }
    });

    // Display the current slider value
    document.getElementById("slider-value").innerText = year;
}
