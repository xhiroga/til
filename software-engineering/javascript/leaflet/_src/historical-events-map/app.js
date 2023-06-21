const map = L.map('map').setView([51.505, -0.09], 2);
let geoJsonData = null;

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

fetch('data.geojson')
    .then(response => response.json())
    .then(data => {
        geoJsonData = data;
        updateMap(document.getElementById("slider").value);
    });

const geojsonLayer = L.geoJson(null, {
    pointToLayer: function (feature, latlng) {
        if (feature.properties.image) {
            var icon = L.icon({
                iconUrl: feature.properties.image,
                iconSize: [38, 38], // size of the icon
                iconAnchor: [22, 94], // point of the icon which will correspond to marker's location
                popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
            });
            return L.marker(latlng, { icon: icon });
        } else {
            return L.marker(latlng);
        }
    }
}).addTo(map);

function updateMap(startYear, endYear) {
    if (!geoJsonData) return;

    // Filter the GeoJSON data based on the year
    const filteredData = {
        ...geoJsonData,
        features: geoJsonData.features.filter(feature => feature.properties.year >= parseInt(startYear) && feature.properties.year <= parseInt(endYear))
    };

    geojsonLayer.clearLayers();
    geojsonLayer.addData(filteredData);
}

const slider = document.getElementById('slider');

noUiSlider.create(slider, {
    start: [1850, 1850],
    connect: true,
    step: 1,
    range: {
        'min': 1850,
        'max': 1950
    }
});

slider.noUiSlider.on('update', function (values, handle) {
    document.getElementById('slider-start-value').innerText = Math.round(values[0]);
    document.getElementById('slider-end-value').innerText = Math.round(values[1]);
    updateMap(values[0], values[1]);
});
