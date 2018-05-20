import React from 'react';
import PropTypes from 'prop-types';
import { Button, StyleSheet, Text, View, ViewPropTypes } from 'react-native';
import { Actions } from 'react-native-router-flux';

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'transparent',
    },
});

class DrawerContent extends React.Component {
    static propTypes = {
        name: PropTypes.string,
        sceneStyle: ViewPropTypes.style,
        title: PropTypes.string,
    }

    static contextTypes = {
        drawer: PropTypes.object,
    }

    render() {
        return (
            <View style={styles.container}>
                <Button onPress={Actions.pop} title="Back"></Button>
                <Button onPress={Actions.threeComponent} title="Three.js"></Button>
                <Button onPress={Actions.camera} title="Camera"></Button>
            </View >
        );
    }
}

export default DrawerContent;