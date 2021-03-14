//(function(){

var firebaseConfig = {
  apiKey: "API_KEY",
  authDomain: "hiroga-gs-with-cloud-firestore.firebaseapp.com",
  projectId: "hiroga-gs-with-cloud-firestore",
  storageBucket: "hiroga-gs-with-cloud-firestore.appspot.com",
  messagingSenderId: "MESSAGING_SENDER_ID",
  appId: "APP_ID",
};
firebase.initializeApp(firebaseConfig);

const firestore = firebase.firestore();
const docRef = firestore.collection("samples").doc("sandwichData");
const outputHeader = document.querySelector("#hotDogOutput");
const inputTextField = document.querySelector("#latestHotDogStatus");
const saveButton = document.querySelector("#saveButton");
const loadButton = document.querySelector("#loadButton");

saveButton.addEventListener("click", function () {
  const textToSave = inputTextField.value;
  console.log("I am going to save " + textToSave + " to firebase");
  docRef
    .set({
      hotDogStatus: textToSave,
    })
    .then(function () {
      console.log("Status saved!");
    })
    .catch(function (error) {
      console.log("Got an error: ", error);
    });
});

loadButton.addEventListener("click", function () {
  docRef
    .get()
    .then(function (doc) {
      if (doc && doc.exists) {
        const myData = doc.data();
        outputHeader.innerText = "Hot dog status: " + myData.hotDogStatus;
      }
    })
    .catch(function (error) {
      console.log("Got an error: ", error);
    });
});

getRealtimeUpdates = function () {
  docRef.onSnapshot(function (doc) {
    if (doc && doc.exists) {
      const myData = doc.data();
      console.log("Check out this document I received", doc);
      outputHeader.innerText = "Hot dog status: " + myData.hotDogStatus;
    }
  });
};

getRealtimeUpdates();

// })
