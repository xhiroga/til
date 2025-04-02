import { _decorator, Component, Node, Collider2D, Contact2DType, IPhysics2DContact, RigidBody2D, Label, director, PhysicsSystem2D } from 'cc'; // Added Label, director, PhysicsSystem2D

const { ccclass, property } = _decorator;

@ccclass('HeroController')
export class HeroController extends Component {

    @property(Node) // Expose a Node property to link the Label in the editor
    gameClearLabel: Node | null = null;

    private heroCollider: Collider2D | null = null;
    private gameIsOver: boolean = false; // Prevent multiple triggers

    onLoad() {
        console.log("HeroController loaded!");
        this.gameIsOver = false; // Reset game state on load
        if (this.gameClearLabel) {
            this.gameClearLabel.active = false; // Ensure label is hidden at start
        } else {
            console.warn("Game Clear Label node is not assigned in the HeroController inspector!");
        }

        this.heroCollider = this.getComponent(Collider2D);
        if (this.heroCollider) {
            // Register the collision callback
            this.heroCollider.on(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);
        } else {
            console.error("Hero node is missing a Collider2D component!");
        }

        // Ensure the Hero has a RigidBody2D component as well (Static or Kinematic recommended if it doesn't move)
        if (!this.getComponent('cc.RigidBody2D')) {
             console.warn("Hero node might need a RigidBody2D component for collisions to register reliably.");
        }
    }

    onDestroy() {
        if (this.heroCollider) {
            // Unregister the callback
            this.heroCollider.off(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);
        }
    }

    onBeginContact(selfCollider: Collider2D, otherCollider: Collider2D, contact: IPhysics2DContact | null) {
        if (this.gameIsOver) return; // Don't trigger multiple times

        console.log("onBeginContact called!");
        
        // Check if the other collider belongs to the 'Treasure' node
        if (otherCollider.node.name === 'Treasure') {
            this.gameIsOver = true; // Set flag
            console.log("GAME CLEAR! Treasure contacted Hero.");

            // Activate the Game Clear Label
            if (this.gameClearLabel) {
                this.gameClearLabel.active = true;
            } else {
                 console.warn("Game Clear Label node is not assigned in the HeroController inspector!");
            }

            // Stop physics simulation
            PhysicsSystem2D.instance.enable = false;

            // Optional: Add other game clear actions (sound, scene change after delay, etc.)

        }
         // Optional: Check for collision with Lava
         else if (otherCollider.node.name === 'Lava') {
             this.gameIsOver = true; // Set flag
             console.log("GAME OVER! Hero contacted Lava.");

             // Stop physics simulation
             PhysicsSystem2D.instance.enable = false;

             // --- Game Over Logic ---
             // Add game over actions here (e.g., show game over screen, restart level after delay)
             // Example: Restart level after 2 seconds
             // this.scheduleOnce(() => {
             //     director.getPhysicsManager().enabled = true; // Re-enable physics before reload
             //     director.loadScene(director.getScene().name);
             // }, 2);
         }
    }
}