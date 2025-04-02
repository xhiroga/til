import { _decorator, Component, Node, input, Input, EventTouch, RigidBody2D, Collider2D, Vec2, tween, v3 } from 'cc';

const { ccclass, property } = _decorator;

@ccclass('PinController')
export class PinController extends Component {

    private body: RigidBody2D | null = null;
    private collider: Collider2D | null = null;
    private isPulled: boolean = false;
    private initialPosition: Vec2 = new Vec2(); // Store initial position if needed for reset or animation

    onLoad() {
        this.body = this.getComponent(RigidBody2D);
        this.collider = this.getComponent(Collider2D);
        if (this.node) {
             // Store initial position based on the node's world position
            const worldPos = this.node.getWorldPosition();
            this.initialPosition.set(worldPos.x, worldPos.y);
            // Register touch/mouse down event listener
            this.node.on(Input.EventType.TOUCH_START, this.onPinClicked, this);
            // Alternatively, for mouse clicks:
            // this.node.on(Input.EventType.MOUSE_DOWN, this.onPinClicked, this);
        }
    }

    onDestroy() {
        // Clean up listeners when the node is destroyed
        if (this.node) {
            this.node.off(Input.EventType.TOUCH_START, this.onPinClicked, this);
            // this.node.off(Input.EventType.MOUSE_DOWN, this.onPinClicked, this);
        }
    }

    onPinClicked(event: EventTouch) {
        if (this.isPulled) {
            return; // Already pulled
        }
        console.log(`Pin ${this.node.name} clicked!`);
        this.isPulled = true;

        // --- Pin Removal Logic ---
        // Option 1: Disable collider immediately
        if (this.collider) {
            this.collider.enabled = false;
        }
        // Option 2: Change RigidBody type (if you want it to fall)
        // if (this.body) {
        //     this.body.type = RigidBody2D.Type.Dynamic;
        // }

        // Option 3: Animate the pin moving out (using tween)
        tween(this.node)
            .to(0.3, { position: v3(this.node.position.x + 200, this.node.position.y, this.node.position.z) }, { easing: 'sineOut' }) // Adjust direction/distance
            .call(() => {
                // Optional: Disable node or collider after animation
                // this.node.active = false;
                if (this.collider) {
                     this.collider.enabled = false; // Ensure collider is off after moving
                }
                if (this.body) {
                    // Ensure physics body doesn't interfere after moving
                    this.body.enabled = false;
                }
            })
            .start();

        // TODO: Add sound effects or visual feedback
    }
}