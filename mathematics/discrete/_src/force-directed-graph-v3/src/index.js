import { ForceDirectedGraphChart } from "chartjs-chart-graph";

fetch("./miserables.json")
  .then((r) => r.json())
  .then((data) => {
    new ForceDirectedGraphChart(
      document.getElementById("canvas").getContext("2d"),
      {
        data: {
          labels: data.nodes.map((d) => d.id),
          datasets: [
            {
              pointBackgroundColor: "steelblue",
              pointRadius: 5,
              data: data.nodes,
              edges: data.links,
            },
          ],
        },
        options: {
          legend: {
            display: false,
          },
        },
      }
    );
  });
