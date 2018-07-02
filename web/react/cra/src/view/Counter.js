import React, { Component } from 'react';

class Counter extends Component {

    constructor(props) {
        super(props)
        this.state = { count: 1 }
    }

    render() {
        return (
            <div className="Counter">
                <p>{this.state.count}</p>
            </div>
        );
    }
}

export default Counter;