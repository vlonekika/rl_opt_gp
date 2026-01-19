from fastapi import FastAPI
from typing import Dict
import logging
from datetime import datetime
import pickle

from catboost import CatBoostClassifier, Pool
from app.ml_tools import state_fe_standart, reward
from app.ab_user_splitter import user_splitter

from app.models import InitEvent, UserSnapshotActiveState, RewardEvent, AdRewardResponse
from app.rl_agent import RLAgent

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="RL Ad Reward Optimization Service",
    description="Reinforcement Learning service for optimizing ad rewards in mobile game",
    version="1.0.0"
)

# Multi-Armed Bandit для оптимизации коэффициента награды за рекламу
# Обучается на каждом единичном REWARD событии (CLICKED/IGNORED)
mab_agent = RLAgent(
    coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    epsilon=0.1,
    epsilon_decay=0.999,
    min_epsilon=0.01,
    penalty_weight=0.1
)

with open("app/ml_models_pkl/ad_model_drop_device.pkl", "rb") as file:
    ad_prob_model = pickle.load(file)

ad_prob_model_features = ad_prob_model.feature_names_

# Хранилище init_data для uplift группы: session_id -> init_event_data
session_init_data: Dict[int, Dict] = {}

GROUPS = ["default", "mab", "uplift"]
SALT = "v1"


@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "Multi-Armed Bandit Ad Reward Optimization",
        "status": "running",
        "version": "1.0.0",
        "mab_stats": mab_agent.get_stats()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/events/init", response_model=AdRewardResponse)
async def handle_init_event(event: InitEvent):
    """
    Обрабатывает init_event - начало новой игровой сессии.
    Возвращает начальную награду за рекламу на основе дефолтного значения.
    """
    logger.info(f"Init event received for session {event.session_id}, device {event.appmetrica_device_id}")

    split_group_id = user_splitter(
        user_id=event.appmetrica_device_id,
        n_buckets=len(GROUPS),
        salt=SALT,
    )
    reward_source = GROUPS[split_group_id]

    # Сохраняем init_data для uplift группы
    if reward_source == "uplift":
        key = str(event.appmetrica_device_id) + "_" + str(event.session_id)
        session_init_data[key] = event.model_dump()
        #session_init_data[event.session_id] = event.model_dump()
        print(session_init_data)

    if reward_source == "mab":
        # MAB выбирает коэффициент
        coefficient = mab_agent.select_action()

        # Дефолтное значение награды (будет обновлено при первом snapshot)
        default_reward = 1000
        recommended_reward = int(coefficient * default_reward)

        logger.info(
            f"Session {event.session_id} initialized: coefficient={coefficient}, "
            f"default_reward={default_reward}, recommended_reward={recommended_reward}"
        )

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="mab",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )
    
    elif reward_source == "uplift":
        default_reward = 1000
        coefficient = 1
        recommended_reward = int(coefficient * default_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )
    
    elif reward_source == "default":
        default_reward = 1000
        coefficient = 1
        recommended_reward = int(coefficient * default_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="default",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )


@app.post("/events/snapshot", response_model=AdRewardResponse)
async def handle_snapshot_event(event: UserSnapshotActiveState):
    """
    Обрабатывает user_snapshot_active_state - минутный срез состояния игрока.
    Использует MAB агента для определения оптимальной награды за рекламу.
    """
    logger.info(f"Snapshot event received for session {event.session_id}, minute {event.game_minute}")

    split_group_id = user_splitter(
        user_id=event.appmetrica_device_id,
        n_buckets=len(GROUPS),
        salt=SALT,
    )
    reward_source = GROUPS[split_group_id]

    if reward_source == "mab":
        # MAB выбирает коэффициент для следующей минуты
        coefficient = mab_agent.select_action()

        # Рассчитываем рекомендованную награду = coefficient * money_ad_reward_calculate
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        logger.info(
            f"Session {event.session_id}, minute {event.game_minute}: "
            f"coefficient={coefficient}, base_reward={base_reward}, recommended_reward={recommended_reward}"
        )

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="mab",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )
    
    elif reward_source == "uplift":
        # Получаем init_data для uplift модели
        key = str(event.appmetrica_device_id) + "_" + str(event.session_id)
        init_data = session_init_data.get(key, {})
        print(init_data)
        state = event.model_dump() | init_data
        fe_state = state_fe_standart(state)

        prob = ad_prob_model.predict_proba(
            Pool(
                [[fe_state.get(f) for f in ad_prob_model_features]],
                feature_names=ad_prob_model_features
            )
        )[:, 1][0]

        coefficient = reward(prob)
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )
    
    elif reward_source == "default":

        coefficient = 1
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="default",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )



@app.post("/events/reward")
async def handle_reward_event(event: RewardEvent):
    """
    Обрабатывает reward event - события рекламы (CLICKED/IGNORED).

    CLICKED - пользователь принял оффер и посмотрел рекламу
    IGNORED - пользователь не принял оффер на просмотр рекламы

    Обучает MAB агента на основе полученного коэффициента и результата.
    """
    logger.info(
        f"Reward event received: session {event.session_id}, "
        f"type {event.event_type}, source {event.reward_source}, "
        f"coefficient {event.recommended_coefficient}, reward {event.recommended_reward}"
    )

    if event.reward_source == "mab":
        # Обучаем MAB на основе результата
        clicked = (event.event_type == "CLICKED")
        mab_agent.update(event.recommended_coefficient, clicked)

        return {
            "status": "ok",
            "session_id": event.session_id,
            "event_type": event.event_type,
            "mab_updated": True
        }

    return {
        "status": "ok",
        "session_id": event.session_id,
        "event_type": event.event_type,
        "mab_updated": False
    }


@app.get("/agent/stats")
async def get_agent_stats():
    """Возвращает статистику MAB агента"""
    return mab_agent.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)