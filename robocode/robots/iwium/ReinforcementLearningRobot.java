package iwium;

import robocode.*;
import java.awt.Color;
import com.github.chen0040.rl.learning.qlearn.QLearner;
import java.lang.Math;
import java.util.Map;
import java.util.HashMap;
import java.util.Random;
import robocode.util.*;
import java.awt.geom.*;

import static robocode.util.Utils.normalRelativeAngle;


//SOURCES: 
//https://github.com/krris/QLearning-Robocode/blob/master/src/main/java/io/github/krris/qlearning/LearningRobot.java
//https://github.com/stevenpjg/QlearningRobocodeNN/blob/master/LookUpTable/Rl_check.java
//https://github.com/DulingLai/Robocode_MLProject/blob/master/src/bots/RL_robot.java

public class ReinforcementLearningRobot extends AdvancedRobot {
	
	private static double gamma = 0.95;
	private static double alpha = 0.80;
	private static double epsilon = 0.10;
	
	private static boolean learning = true;
	private static int statesCount = 3 * 3 * 3 * 4 * 2 * 9;
	private static int stateComponents = 6;
	private static int actionsCount = 6;
	
	private static final QLearner agent = new QLearner(statesCount, actionsCount, alpha, gamma, 0.5);
	
	private int previousState;
	private int currentState;
	
	private Action previousAction;
	private Action currentAction;
	
	private double sumReward;
	
	private double arenaHeight;
	private double arenaWidth;
	
	private Mode mode;
	
	private Map<String, Integer> rewards;
	
	private enum Action {
		MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, FIRE_SIMPLE, FIRE_CIRCULAR;
		
		public static double moveDistance = 30;
		public static int fireStrength = 2;
		
		public static Action fromInt(int a) {
			if(a == 0) return MOVE_UP;
			if(a == 1) return MOVE_DOWN;
			if(a == 2) return MOVE_LEFT;
			if(a == 3) return MOVE_RIGHT;
			if(a == 4) return FIRE_SIMPLE;
			return FIRE_CIRCULAR;
		}
		
		public int toInt() {
			if(this.equals(MOVE_UP)) return 0;
			if(this.equals(MOVE_DOWN)) return 1;
			if(this.equals(MOVE_LEFT)) return 2;
			if(this.equals(MOVE_RIGHT)) return 3;
			if(this.equals(FIRE_SIMPLE)) return 4;
			return 5;
		}
	}
	
	private double enemyEnergy;
	private double enemyVelocity;
	private double enemyBearing;
	private double enemyHeading;
	private double enemyDistance;
	
	private ScannedRobotEvent lastScan;
	
	private double oldEnemyHeading;
	
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
			//execute();
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
		previousState = currentState;
		previousAction = currentAction;
		currentState = getStateId();
		if(learning) {
			updateAgent();
		}
		int action;
		if(Math.random() < epsilon) {
			action = selectRandomAction();
		}
		else {
			action = agent.selectAction(currentState).getIndex();
		}
		currentAction = Action.fromInt(action);
		lastScan = null;
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
			   3 * 3 * 3 * 4 * getEnemyMovementId() +
			   3 * 3 * 3 * 4 * 2 * getLocationId();
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
		double angle = 90;
		for(double langle = -45; langle < 315; langle += angle) {
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
		if(x > 3.0 * arenaWidth / 4.0) {
			return 2;
		}
		return 1;
	}
	
	private int getLocationYId() {
		double y = getY();
		if(y < arenaHeight / 4.0) {
			return 0;
		}
		if(y > 3.0 * arenaHeight / 4.0) {
			return 2;
		}
		return 1;
	}
	
	private int selectRandomAction() {
		return new Random().nextInt(actionsCount);
	}
	
	private void act() {
		if(currentAction.equals(Action.MOVE_UP)) {
			double heading = getHeading();
			if(heading >= 180) {
				setTurnRight(360 - heading);
			}
			else {
				setTurnLeft(heading);
			}
			setAhead(Action.moveDistance);
			execute();
		}
		else if(currentAction.equals(Action.MOVE_DOWN)) {
			double heading = getHeading();
			if(heading >= 180) {
				setTurnLeft(180 - heading);
			}
			else {
				setTurnRight(heading - 180);
			}
			setAhead(Action.moveDistance);
			execute();
		}
		else if(currentAction.equals(Action.MOVE_LEFT)) {
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
			execute();
		}
		else if(currentAction.equals(Action.MOVE_RIGHT)) {
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
			execute();
		}
		else if(currentAction.equals(Action.FIRE_SIMPLE)) {
			fireLinear();
		}
		else if(currentAction.equals(Action.FIRE_CIRCULAR)) {
			fireCircular();
		}
	}
	
	private void fireLinear() {
		double turn = normalRelativeAngle(enemyBearing + getHeadingRadians() - getGunHeadingRadians());
		setTurnGunRightRadians(turn);
		setFire(Action.fireStrength);
		execute();
	}
	
	private void fireCircular() {
		if(lastScan == null) {
			fireLinear();
			return;
		}
		double _bulletPower = Action.fireStrength;
		double _myX = getX();
		double _myY = getY();
		double _absoluteBearing = getHeadingRadians() + lastScan.getBearingRadians();
		double _enemyX = getX() + lastScan.getDistance() * Math.sin(_absoluteBearing);
		double _enemyY = getY() + lastScan.getDistance() * Math.cos(_absoluteBearing);
		double _enemyHeading = lastScan.getHeadingRadians();
		double _enemyHeadingChange = enemyHeading - oldEnemyHeading;
		double _enemyVelocity = lastScan.getVelocity();
		oldEnemyHeading = enemyHeading;

		double deltaTime = 0;
		double battleFieldHeight = getBattleFieldHeight(), battleFieldWidth = getBattleFieldWidth();
		double predictedX = _enemyX, predictedY = _enemyY;
		while((++deltaTime) * (20.0 - 3.0 * _bulletPower) < Point2D.Double.distance(_myX, _myY, predictedX, predictedY)){		
			predictedX += Math.sin(_enemyHeading) * _enemyVelocity;
			predictedY += Math.cos(_enemyHeading) * _enemyVelocity;
			_enemyHeading += _enemyHeadingChange;
			if(predictedX < 18.0 || predictedY < 18.0 || predictedX > battleFieldWidth - 18.0 || predictedY > battleFieldHeight - 18.0) {
				predictedX = Math.min(Math.max(18.0, predictedX), battleFieldWidth - 18.0);	
				predictedY = Math.min(Math.max(18.0, predictedY), battleFieldHeight - 18.0);
				break;
			}
		}
		double theta = Utils.normalAbsoluteAngle(Math.atan2(predictedX - getX(), predictedY - getY()));
		setTurnRadarRightRadians(Utils.normalRelativeAngle(_absoluteBearing - getRadarHeadingRadians()));
		setTurnGunRightRadians(Utils.normalRelativeAngle(theta - getGunHeadingRadians()));
		fire(Action.fireStrength);
	}
	
	private void initializeRobot() {
		setBodyColor(Color.red);
		setGunColor(Color.black);
		setRadarColor(Color.yellow);
		setBulletColor(Color.green);
		setScanColor(Color.green);

		setAdjustGunForRobotTurn(true);
		setAdjustRadarForGunTurn(true);
		
		mode = Mode.SENSE;
		
		currentAction = Action.FIRE_SIMPLE;
		currentState = 0;
		oldEnemyHeading = 0;
	}
	
	private void initializeAgent() {
		arenaWidth = getBattleFieldWidth();
		arenaHeight = getBattleFieldHeight();
		
		sumReward = 0.0;
		
		previousState = -1;
		currentState = -1;
		
		rewards = new HashMap<String, Integer>();
		rewards.put("hitByBullet", -10);
		rewards.put("hitWall", -5);
		rewards.put("hitRobot", 1);
		rewards.put("bulletHit", 10);
		rewards.put("death", -100);
		rewards.put("kill", 75);
		rewards.put("bulletMissed", -5);
		
	}

	public void onScannedRobot(ScannedRobotEvent event) {
		enemyDistance = event.getDistance();
		enemyBearing = event.getBearingRadians();
		enemyHeading = event.getHeadingRadians();
		enemyVelocity = event.getVelocity();
		enemyEnergy = event.getEnergy();
		
		lastScan = event;
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
		if(learning) {
			updateAgent();
		}
	}
	
	public void onRobotDeath(RobotDeathEvent event) {
		sumReward += rewards.get("kill");
	}
	
	public void onBulletMissed(BulletMissedEvent event) {
		sumReward += rewards.get("bulletMissed");
	}
	
	public void onRoundEnded(RoundEndedEvent event) {
		
	}
	
	public void onBattleEnded(BattleEndedEvent event) {
		
	}
	
}
