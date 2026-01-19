import numpy as np
from typing import Dict
import logging
import threading

logger = logging.getLogger(__name__)


class MultiArmedBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit для оптимизации коэффициента награды за рекламу.
    Каждая "рука" (arm) соответствует коэффициенту к money_ad_reward_calculate.

    Обучается на каждом единичном REWARD событии (CLICKED/IGNORED).
    """

    def __init__(
        self,
        coefficients: list = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        penalty_weight: float = 0.1
    ):
        # Коэффициенты награды (arms)
        if coefficients is None:
            coefficients = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        self.arms = coefficients
        self.n_arms = len(self.arms)

        # Вес штрафа за высокие награды (чем выше, тем сильнее штраф)
        self.penalty_weight = penalty_weight

        # Статистика для каждой руки
        # arm_id -> {count: int, total_reward: float, avg_reward: float}
        self.arm_stats = {arm: {'count': 0, 'total_reward': 0.0, 'avg_reward': 0.0} for arm in self.arms}

        # Гиперпараметры
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Статистика
        self.total_pulls = 0
        self.total_rewards = 0.0

        # Thread safety для конкурентных обновлений от нескольких игроков
        self._lock = threading.Lock()

        logger.info(
            f"Multi-Armed Bandit initialized with {self.n_arms} coefficient arms: "
            f"{self.arms}, penalty_weight={penalty_weight}"
        )

    def select_action(self, exploit_only: bool = False) -> float:
        """
        Выбирает коэффициент награды используя epsilon-greedy стратегию.

        Args:
            exploit_only: Если True, использует только exploitation (без exploration)

        Returns:
            Коэффициент для умножения на money_ad_reward_calculate
        """
        # Exploration: случайный выбор коэффициента
        if not exploit_only and np.random.random() < self.epsilon:
            action = float(np.random.choice(self.arms))
            logger.debug(f"Exploration: selected random coefficient {action}")
            return action

        # Exploitation: выбираем коэффициент с максимальной средней наградой
        best_arm = max(self.arms, key=lambda arm: self.arm_stats[arm]['avg_reward'])
        logger.debug(f"Exploitation: selected best coefficient {best_arm} (avg reward: {self.arm_stats[best_arm]['avg_reward']:.3f})")
        return float(best_arm)

    def update(self, coefficient: float, clicked: bool):
        """
        Обновляет статистику выбранного коэффициента на основе единичного события.
        Thread-safe для конкурентных обновлений от нескольких игроков.

        Reward:
        - CLICKED: 1.0 - penalty * coefficient (пользователь посмотрел рекламу)
        - IGNORED: 0.0 - penalty * coefficient (пользователь отклонил)

        Args:
            coefficient: Коэффициент награды, который был предложен
            clicked: True если CLICKED, False если IGNORED
        """
        if coefficient not in self.arms:
            logger.warning(f"Unknown coefficient {coefficient}, skipping update")
            return

        # Рассчитываем reward
        penalty = self.penalty_weight * coefficient
        base_reward = 1.0 if clicked else 0.0
        reward = base_reward - penalty

        # Блокируем доступ к arm_stats для атомарного обновления
        with self._lock:
            # Обновляем статистику коэффициента
            stats = self.arm_stats[coefficient]
            stats['count'] += 1
            stats['total_reward'] += reward
            stats['avg_reward'] = stats['total_reward'] / stats['count']

            # Общая статистика
            self.total_pulls += 1
            self.total_rewards += reward

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            event_type = "CLICKED" if clicked else "IGNORED"
            logger.info(
                f"MAB update: coefficient={coefficient}, event={event_type}, "
                f"penalty={penalty:.2f}, reward={reward:.2f}, "
                f"avg_reward={stats['avg_reward']:.3f}, epsilon={self.epsilon:.3f}"
            )

            if self.total_pulls % 100 == 0:
                logger.info(f"MAB total updates: {self.total_pulls}, epsilon: {self.epsilon:.3f}, avg reward: {self.total_rewards/self.total_pulls:.3f}")

    def get_stats(self) -> Dict:
        """Возвращает статистику агента (thread-safe)"""
        with self._lock:
            # Находим лучшую руку
            best_arm = max(self.arms, key=lambda arm: self.arm_stats[arm]['avg_reward'])
            best_arm_stats = self.arm_stats[best_arm]

            # Топ-5 рук по средней награде
            top_arms = sorted(
                self.arms,
                key=lambda arm: self.arm_stats[arm]['avg_reward'],
                reverse=True
            )[:5]

            return {
                "total_pulls": self.total_pulls,
                "total_rewards": self.total_rewards,
                "avg_reward": self.total_rewards / self.total_pulls if self.total_pulls > 0 else 0.0,
                "epsilon": self.epsilon,
                "n_arms": self.n_arms,
                "best_arm": best_arm,
                "best_arm_count": best_arm_stats['count'],
                "best_arm_avg_reward": best_arm_stats['avg_reward'],
                "top_5_arms": [
                    {
                        "arm": arm,
                        "count": self.arm_stats[arm]['count'],
                        "avg_reward": self.arm_stats[arm]['avg_reward']
                    }
                    for arm in top_arms
                ]
            }


# Алиас для обратной совместимости
RLAgent = MultiArmedBandit