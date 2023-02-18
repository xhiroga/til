import React, { Component } from 'react';
import { Animated, PanResponder, View } from 'react-native';
import Expo from "expo";
import * as THREE from "three";
import ExpoTHREE from "expo-three";


// Expo.GLViewがwebでのcanvasにあたる
export default class ThreeComponent extends Component {
    constructor(props) {
        super(props);

        this.state = {
            pan: new Animated.ValueXY()
        };
    }

    componentWillMount() {
        console.log("componentWillMount FIRE!!");
        this._val = { x: 0, y: 0 };
        this.state.pan.addListener(value => (this._val = value));

        this.panResponder = PanResponder.create({
            onStartShouldSetPanResponder: (e, gesture) => true,
            onPanResponderGrant: (e, gesture) => {
                this.state.pan.setOffset({
                    x: this._val.x,
                    y: this._val.y
                });
                this.state.pan.setValue({ x: 0, y: 0 });
            },
            onPanResponderMove: Animated.event([
                null,
                { dx: this.state.pan.x, dy: this.state.pan.y }
            ])
        });
    }

    render() {
        const panStyle = {
            transform: this.state.pan.getTranslateTransform()
        };
        return (
            <View>
                <Animated.View
                    {...this.panResponder.panHandlers}
                    style={[panStyle, { width: 200, height: 200 }]}
                >
                    <Expo.GLView
                        style={{ flex: 1 }}
                        onContextCreate={this._onGLContextCreate}
                    />
                </Animated.View>
            </View>
        )
    }

    // Expo.GLViewをターゲットとしてThree.jsのレンダリングをするためにExpoTHREEが必要
    _onGLContextCreate = async gl => {
        // console.log(gl); // WebGL2RenderingContextオブジェクトが返却される。
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
            75, gl.drawingBufferWidth / gl.drawingBufferHeight, 0.1, 1000
        );
        const renderer = ExpoTHREE.createRenderer({ gl });
        renderer.setSize(gl.drawingBufferWidth, gl.drawingBufferHeight);

        const geometry = new THREE.SphereBufferGeometry(1, 36, 36);
        const material = new THREE.MeshBasicMaterial({
            map: await ExpoTHREE.createTextureAsync({
                asset: Expo.Asset.fromModule(require("./img/moon_map_mercator.jpg"))
                // Image from David Jeffery Site
                // https://www.nhn.ou.edu/~jeffery/
            })
        });
        const sphere = new THREE.Mesh(geometry, material);
        scene.add(sphere);
        camera.position.z = 2;
        const render = () => {
            requestAnimationFrame(render);
            sphere.rotation.x += 0.01;
            sphere.rotation.y += 0.01;
            renderer.render(scene, camera);
            gl.endFrameEXP();
        };
        render();
    };
}

    // Reference
// https://medium.com/@yoobi55/creating-3d-sphere-component-with-react-native-and-three-c5fb46dadbd