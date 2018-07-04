import React, { Component } from 'react';
import { Text, View } from 'react-native';
import { Drawer, Scene, Router } from 'react-native-router-flux';
import ThreeComponent from './components/ThreeComponent';
import Camera from './components/Camera';
import Firebase from './components/Firebase/Firebase';
import DrawerComponent from './components/drawer/DrawerComponent';

class RouterComponent extends Component {

    // renderScene

    render() {
        return (
            <Router>
                <Drawer key="root" contentComponent={DrawerComponent}>

                    <Scene key="firebase" component={Firebase} init={true} />
                    <Scene key="threeComponent" component={ThreeComponent} />
                    <Scene key="camera" component={Camera} />
                </Drawer>
            </Router>
        );
    }
};


export default RouterComponent;