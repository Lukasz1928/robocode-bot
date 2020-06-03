package iwium;

import robocode.*;
import java.awt.Color;
import com.github.chen0040.rl.learning.qlearn.QLearner;
import java.lang.Math;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;

import static robocode.util.Utils.normalRelativeAngle;


//SOURCES: 
//https://github.com/krris/QLearning-Robocode/blob/master/src/main/java/io/github/krris/qlearning/LearningRobot.java
//https://github.com/stevenpjg/QlearningRobocodeNN/blob/master/LookUpTable/Rl_check.java
//https://github.com/DulingLai/Robocode_MLProject/blob/master/src/bots/RL_robot.java

public class ReinforcementLearningRobot extends AdvancedRobot {
	
	private double gamma = 0.95;
	private double alpha = 0.80;
	private double epsilon = 0.10;
	
	private QLearner agent; 
	
	private boolean learning = true;
	private int statesCount = 3 * 3 * 3 * 8 * 2 * 9;
	private int stateComponents = 6;
	private int actionsCount = 5;
	
	private int previousState;
	private int currentState;
	
	private Action previousAction;
	private Action currentAction;
	
	private double sumReward;
	
	private double arenaHeight;
	private double arenaWidth;
	
	private Mode mode;
	
	private Action nextAction;
	
	private Map<String, Integer> rewards;
	
	private enum Action {
		MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, FIRE;
		
		public static double moveDistance = 10;
		public static int fireStrength = 2;
		
		public static Action fromInt(int a) {
			if(a == 0) return MOVE_UP;
			if(a == 1) return MOVE_DOWN;
			if(a == 2) return MOVE_LEFT;
			if(a == 3) return MOVE_RIGHT;
			return FIRE;
		}
		
		public int toInt() {
			if(this.equals(MOVE_UP)) return 0;
			if(this.equals(MOVE_DOWN)) return 1;
			if(this.equals(MOVE_LEFT)) return 2;
			if(this.equals(MOVE_RIGHT)) return 3;
			return 4;
		}
	}
	
	private boolean scanned;
	private double enemyEnergy;
	private double enemyVelocity;
	private double enemyBearing;
	private double enemyHeading;
	private double enemyDistance;
	
	private enum Mode {
		SENSE, THINK, ACT;
		
		public Mode next() {
			if(this.equals(SENSE)) {
				return THINK;
			}
			if(this.equals(THINK)) {
				return ACT;
			}
			return SENSE;
		}
	}

	public void run() {
		initializeRobot();
		initializeAgent();

		while(true) {
			if(mode.equals(Mode.SENSE)) {
				sense();
			}
			if(mode.equals(Mode.THINK)) {
				think();
			}
			if(mode.equals(Mode.ACT)) {
				act();
			}
			execute();
			mode = mode.next();
		}
	}
	
	private void sense() {
		if(getAllEvents().isEmpty()) {
			setTurnRadarRight(90);
			execute();
		}
	}
	
	private void think() {
		if(learning) {
			updateAgent();
		}
		previousState = currentState;
		previousAction = currentAction;
		currentState = getStateId();
		int action;
		if(Math.random() < epsilon) {
			action = selectRandomAction();
		}
		else {
			action = agent.selectAction(currentState).getIndex();
		}
		currentAction = Action.fromInt(action);
	}
	
	private void updateAgent() {
		agent.update(previousState, previousAction.toInt(), currentState, sumReward);
		sumReward = 0.0;
	}
	
	private int getStateId() {
		return 1 * getEnergyId() +
			   3 * getEnemyEnergyId() +
			   3 * 3 * getEnemyDistanceId() +
			   3 * 3 * 3 * getEnemyAngleId() +
			   3 * 3 * 3 * 8 * getEnemyMovementId() +
			   3 * 3 * 3 * 8 * 2 * getLocationId();
	}
	
	private int getEnergyId() {
		double energy = getEnergy();
		if(energy < 10) {
			return 0;
		}
		if(energy < 50) {
			return 1;
		}
		return 2;
	}
	
	private int getEnemyEnergyId() {
		if(enemyEnergy <= 8) {
			return 0;
		}
		if(enemyEnergy <= 40) {
			return 1;
		}
		return 2;
	}
	
	private int getEnemyDistanceId() {
		if(enemyDistance <= 20) {
			return 0;
		}
		if(enemyDistance <= 100) {
			return 1;
		}
		return 2;
	}
	
	private int getEnemyAngleId() {
		int id = 0;
		double angle = 45;
		for(double langle = -22.5; langle < 337.5; langle += angle) {
			if(enemyBearing > langle && enemyBearing <= langle + angle) {
				return id;
			}
			id++;
		}
		return 0;
	}
	
	private int getEnemyMovementId() {
		if(enemyVelocity < 1) {
			return 0;
		}
		return 1;
	}
	
	private int getLocationId() {
		return 3 * getLocationXId() + getLocationYId();
	}
	
	private int getLocationXId() {
		double x = getX();
		if(x < arenaWidth / 4.0) {
			return 0;
		}
		if(x < 3.0 * arenaWidth / 4.0) {
			return 2;
		}
		return 1;
	}
	
	private int getLocationYId() {
		double y = getY();
		if(y < arenaWidth / 4.0) {
			return 0;
		}
		if(y < 3.0 * arenaWidth / 4.0) {
			return 2;
		}
		return 1;
	}
	
	private int selectRandomAction() {
		return new Random().nextInt(this.actionsCount);
	}
	
	private void act() {
		if(nextAction.equals(Action.MOVE_UP)) {
			double heading = getHeading();
			if(heading >= 180) {
				setTurnRight(360 - heading);
			}
			else {
				setTurnLeft(heading);
			}
			setAhead(Action.moveDistance);
		}
		else if(nextAction.equals(Action.MOVE_DOWN)) {
			double heading = getHeading();
			if(heading >= 180) {
				setTurnLeft(180 - heading);
			}
			else {
				setTurnRight(heading - 180);
			}
			setAhead(Action.moveDistance);
		}
		else if(nextAction.equals(Action.MOVE_LEFT)) {
			double heading = getHeading();
			if(heading <= 90) {
				setTurnLeft(90 + heading);
			}
			else if(heading <= 270) {
				setTurnRight(270 - heading);
			}
			else {
				setTurnLeft(heading - 270);
			}
			setAhead(Action.moveDistance);
		}
		else if(nextAction.equals(Action.MOVE_RIGHT)) {
			double heading = getHeading();
			if(heading < 90) {
				setTurnRight(90 - heading);
			}
			else if(heading < 270) {
				setTurnLeft(heading - 90);
			}
			else {
				setTurnRight(450 - heading);
			}
			setAhead(Action.moveDistance);
		}
		else if(nextAction.equals(Action.FIRE)) {
			fire();
		}
	}
	
	private void fire() {
		double turn = normalRelativeAngle(enemyBearing + getHeadingRadians() - getGunHeadingRadians());
		setTurnGunRightRadians(turn);
		setFire(Action.fireStrength);
	}
	
	private void initializeRobot() {
		setColors(Color.red,Color.blue,Color.green);

		setAdjustGunForRobotTurn(true);
		setAdjustRadarForGunTurn(true);
		
		mode = Mode.SENSE;
		
	}
	
	private void initializeAgent() {
		arenaWidth = getBattleFieldWidth();
		arenaHeight = getBattleFieldHeight();
		
		sumReward = 0.0;
		
		agent = new QLearner();
		agent.getModel().setAlpha(this.alpha);
		agent.getModel().setGamma(this.gamma);
		
		previousState = -1;
		currentState = -1;
		
		rewards = new HashMap<String, Integer>();
		rewards.put("hitByBullet", -10);
		rewards.put("hitWall", -5);
		rewards.put("hitRobot", -3);
		rewards.put("bulletHit", 10);
		rewards.put("death", -100);
		rewards.put("kill", 75);
		
	}

	public void onScannedRobot(ScannedRobotEvent event) {
		enemyDistance = event.getDistance();
		enemyBearing = event.getBearingRadians();
		enemyHeading = event.getHeadingRadians();
		enemyVelocity = event.getVelocity();
		enemyEnergy = event.getEnergy();
		
	}

	public void onHitWall(HitWallEvent event) {
		sumReward += rewards.get("hitWall");
	}
	
	public void onHitRobot(HitRobotEvent event) {
		sumReward += rewards.get("hitRobot");
	}
	
	public void onBulletHit(BulletHitEvent event) {
		sumReward += rewards.get("bulletHit");
	}
	
	public void onHitByBullet(HitByBulletEvent event) {
		sumReward += rewards.get("hitByBullet");
	}
	
	public void onDeath(DeathEvent event) {
		sumReward += rewards.get("death");
		updateAgent();
	}
	
	public void onRobotDeath(RobotDeathEvent event) {
		sumReward += rewards.get("kill");
	}
	
	public void onRoundEnded(RoundEndedEvent event) {
		
	}
	
	public void onBattleEnded(BattleEndedEvent event) {
		
	}
	
}
